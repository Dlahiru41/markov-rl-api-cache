# Implementation Reference: Code Examples
## Markov RL API Cache - Ready-to-Use Code

---

## Table of Contents

1. [FastAPI Integration](#fastapi-integration)
2. [Django Integration](#django-integration)
3. [Async Python Integration](#async-python-integration)
4. [Go/Node.js Backend Communication](#gonode-backend-communication)
5. [Docker Compose Setup](#docker-compose-setup)
6. [Kubernetes Manifests](#kubernetes-manifests)
7. [Monitoring & Metrics](#monitoring--metrics)

---

## FastAPI Integration

### Complete FastAPI Service with Caching

```python
# main_fastapi.py
"""
Complete FastAPI service integrated with Markov RL Cache
"""

from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import httpx
import time
import logging
import json
from datetime import datetime
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== CONFIGURATION =====
CACHE_SERVICE_URL = "http://cache-intelligence:8000"
BACKEND_SERVICES = {
    'products': 'http://products-api:8080',
    'users': 'http://users-api:8080',
    'orders': 'http://orders-api:8080',
}
REQUEST_TIMEOUT = 30

# ===== CACHE CLIENT =====
class CacheClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = None
    
    async def init(self):
        self.client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)
    
    async def close(self):
        if self.client:
            await self.client.aclose()
    
    async def should_serve_from_cache(self, 
                                     endpoint: str, 
                                     user_type: str,
                                     session_history: list) -> Dict[str, Any]:
        """Ask cache service what to do."""
        try:
            response = await self.client.post(
                f"{self.base_url}/decide",
                json={
                    "endpoint": endpoint,
                    "user_type": user_type,
                    "session_history": session_history
                }
            )
            return response.json()
        except Exception as e:
            logger.error(f"Cache decision error: {e}")
            # Fallback: serve from backend
            return {"action": "fetch_from_backend"}
    
    async def get_from_cache(self, key: str) -> Optional[Dict]:
        """Retrieve from cache."""
        try:
            response = await self.client.get(
                f"{self.base_url}/cache/get",
                params={"key": key}
            )
            data = response.json()
            if data.get("hit"):
                return data.get("data")
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
        return None
    
    async def set_in_cache(self, key: str, data: Dict, ttl: int = 3600):
        """Store in cache."""
        try:
            await self.client.post(
                f"{self.base_url}/cache/set",
                json={"key": key, "data": data, "ttl_seconds": ttl}
            )
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
    
    async def record_metrics(self, endpoint: str, user_id: str, 
                            user_type: str, response_time_ms: float,
                            cache_hit: bool):
        """Record metrics for model training."""
        try:
            await self.client.post(
                f"{self.base_url}/metrics/record",
                json={
                    "endpoint": endpoint,
                    "user_id": user_id,
                    "user_type": user_type,
                    "response_time_ms": response_time_ms,
                    "cache_hit": cache_hit,
                    "timestamp": datetime.now().isoformat()
                }
            )
        except Exception as e:
            logger.warning(f"Metrics error: {e}")

# ===== STARTUP/SHUTDOWN =====
cache_client = CacheClient(CACHE_SERVICE_URL)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    # Startup
    await cache_client.init()
    logger.info("Cache client initialized")
    
    # Verify cache service
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{CACHE_SERVICE_URL}/health")
            if response.status_code == 200:
                logger.info("✓ Cache service healthy")
            else:
                logger.warning("⚠ Cache service may be unhealthy")
    except Exception as e:
        logger.warning(f"Cache service not available: {e}")
    
    yield
    
    # Shutdown
    await cache_client.close()
    logger.info("Cache client closed")

# ===== FASTAPI APP =====
app = FastAPI(
    title="API Gateway with Intelligent Caching",
    description="FastAPI service with Markov RL Cache integration",
    version="1.0.0",
    lifespan=lifespan
)

# ===== MIDDLEWARE =====
@app.middleware("http")
async def add_metrics(request: Request, call_next):
    """Add request metrics."""
    request.state.start_time = time.time()
    request.state.user_type = request.headers.get("X-User-Type", "guest")
    request.state.user_id = request.headers.get("X-User-ID", "anonymous")
    request.state.session_id = request.headers.get("X-Session-ID", "unknown")
    
    response = await call_next(request)
    
    # Add timing header
    elapsed_ms = (time.time() - request.state.start_time) * 1000
    response.headers["X-Response-Time"] = f"{elapsed_ms:.1f}ms"
    
    return response

# ===== ENDPOINTS =====

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/{service}/{path:path}")
async def api_endpoint(
    service: str,
    path: str,
    request: Request,
    x_user_type: str = Header(default="guest"),
    x_cache_bypass: bool = Header(default=False)
):
    """
    Main API endpoint with intelligent caching.
    
    Query the cache intelligence system to decide whether to:
    1. Serve from cache
    2. Fetch from backend + cache
    3. Prefetch related data
    """
    
    full_endpoint = f"{service}/{path}"
    cache_key = f"api:{service}:{path}"
    
    logger.info(f"Request: {full_endpoint} (user_type={x_user_type})")
    
    # === STEP 1: Ask cache intelligence what to do ===
    
    if not x_cache_bypass:
        cache_decision = await cache_client.should_serve_from_cache(
            endpoint=full_endpoint,
            user_type=x_user_type,
            session_history=[]  # Could track session history
        )
        
        # === STEP 2: Serve from cache if recommended ===
        if cache_decision.get("action") == "serve_from_cache":
            cached_data = await cache_client.get_from_cache(cache_key)
            if cached_data is not None:
                logger.info(f"Cache HIT: {full_endpoint}")
                return JSONResponse(
                    content=cached_data,
                    headers={"X-Cache": "HIT"}
                )
    
    # === STEP 3: Fetch from backend ===
    backend_start = time.time()
    
    try:
        if service not in BACKEND_SERVICES:
            raise HTTPException(status_code=404, detail=f"Service '{service}' not found")
        
        backend_url = BACKEND_SERVICES[service]
        full_url = f"{backend_url}/{path}"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                full_url,
                timeout=REQUEST_TIMEOUT,
                headers={
                    "X-Forwarded-For": request.client.host if request.client else "unknown",
                }
            )
        
        backend_time_ms = (time.time() - backend_start) * 1000
        backend_data = response.json()
        
        logger.info(f"Cache MISS: {full_endpoint} ({backend_time_ms:.1f}ms)")
        
        # === STEP 4: Cache the response ===
        ttl = cache_decision.get("ttl_seconds", 3600) if not x_cache_bypass else 0
        if ttl > 0:
            await cache_client.set_in_cache(cache_key, backend_data, ttl)
            
            # === STEP 5: Prefetch if recommended ===
            if "prefetch_list" in cache_decision:
                for prefetch_endpoint in cache_decision["prefetch_list"][:3]:
                    # Async prefetch (don't wait)
                    asyncio.create_task(
                        prefetch_endpoint_async(prefetch_endpoint)
                    )
        
        # === STEP 6: Record metrics for model training ===
        await cache_client.record_metrics(
            endpoint=full_endpoint,
            user_id=request.state.user_id,
            user_type=x_user_type,
            response_time_ms=backend_time_ms,
            cache_hit=False
        )
        
        return JSONResponse(
            content=backend_data,
            headers={"X-Cache": "MISS"}
        )
    
    except httpx.RequestError as e:
        logger.error(f"Backend error: {e}")
        raise HTTPException(status_code=502, detail="Backend service unavailable")
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail="Invalid backend response")

@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{CACHE_SERVICE_URL}/cache/stats")
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Cache service error: {e}")

@app.get("/models/info")
async def models_info():
    """Get model information."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{CACHE_SERVICE_URL}/models/info")
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Cache service error: {e}")

# ===== HELPER FUNCTIONS =====

async def prefetch_endpoint_async(endpoint: str):
    """Prefetch endpoint in background."""
    try:
        async with httpx.AsyncClient() as client:
            service, path = endpoint.split('/', 1)
            if service in BACKEND_SERVICES:
                backend_url = BACKEND_SERVICES[service]
                response = await client.get(f"{backend_url}/{path}", timeout=10)
                
                if response.status_code == 200:
                    cache_key = f"api:{endpoint}"
                    await cache_client.set_in_cache(cache_key, response.json())
                    logger.info(f"Prefetched: {endpoint}")
    except Exception as e:
        logger.debug(f"Prefetch failed: {endpoint} - {e}")

# ===== RUN =====
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=4,
        log_level="info"
    )
```

**Usage**:
```bash
# Install dependencies
pip install fastapi uvicorn httpx

# Run with cache intelligence service
docker-compose up -d  # Starts cache intelligence
python main_fastapi.py

# Test
curl http://localhost:8000/api/products/123 \
  -H "X-User-Type: premium"
```

---

## Django Integration

### Django Middleware with Caching

```python
# django_caching.py
"""
Django middleware for Markov RL Cache integration
"""

import httpx
import json
import logging
from django.utils.deprecation import MiddlewareMixin
from django.http import JsonResponse
import time

logger = logging.getLogger(__name__)

CACHE_SERVICE_URL = "http://cache-intelligence:8000"

class IntelligentCacheMiddleware(MiddlewareMixin):
    """
    Django middleware that integrates with Markov RL Cache.
    
    Intercepts requests and decides whether to:
    1. Serve from cache
    2. Fetch from database/backend
    """
    
    def __init__(self, get_response):
        self.get_response = get_response
        self.cache_client = httpx.Client(timeout=30)
        super().__init__(get_response)
    
    def process_request(self, request):
        """Process incoming request."""
        
        # Skip non-GET requests
        if request.method != 'GET':
            return None
        
        # Skip health checks
        if request.path.startswith('/health'):
            return None
        
        # Extract user info
        user_type = self._get_user_type(request)
        user_id = self._get_user_id(request)
        
        request._cache_start_time = time.time()
        request._user_type = user_type
        request._user_id = user_id
        
        # === Ask cache what to do ===
        cache_decision = self._get_cache_decision(
            endpoint=request.path,
            user_type=user_type
        )
        
        request._cache_decision = cache_decision
        
        # === Serve from cache if recommended ===
        if cache_decision.get('action') == 'serve_from_cache':
            cached_response = self._get_from_cache(request.path)
            if cached_response:
                return JsonResponse(
                    cached_response,
                    headers={'X-Cache': 'HIT'}
                )
        
        return None
    
    def process_response(self, request, response):
        """Process response before sending to client."""
        
        # Add response time header
        elapsed_ms = (time.time() - getattr(request, '_cache_start_time', time.time())) * 1000
        response['X-Response-Time'] = f"{elapsed_ms:.1f}ms"
        
        # Cache successful responses
        if response.status_code == 200 and request.method == 'GET':
            cache_decision = getattr(request, '_cache_decision', {})
            ttl = cache_decision.get('ttl_seconds', 3600)
            
            if ttl > 0:
                try:
                    # Get response content
                    if hasattr(response, 'data'):
                        data = response.data
                    else:
                        data = json.loads(response.content)
                    
                    self._set_in_cache(request.path, data, ttl)
                except Exception as e:
                    logger.warning(f"Failed to cache response: {e}")
        
        # Record metrics
        self._record_metrics(request, response, elapsed_ms)
        
        return response
    
    # === HELPER METHODS ===
    
    def _get_user_type(self, request) -> str:
        """Extract user type from request."""
        return request.headers.get('X-User-Type', 'guest')
    
    def _get_user_id(self, request) -> str:
        """Extract user ID from request."""
        if request.user.is_authenticated:
            return str(request.user.id)
        return request.headers.get('X-User-ID', 'anonymous')
    
    def _get_cache_decision(self, endpoint: str, user_type: str) -> dict:
        """Ask cache intelligence what to do."""
        try:
            response = self.cache_client.post(
                f"{CACHE_SERVICE_URL}/decide",
                json={
                    "endpoint": endpoint,
                    "user_type": user_type
                }
            )
            return response.json()
        except Exception as e:
            logger.warning(f"Cache decision error: {e}")
            return {"action": "fetch_from_backend"}
    
    def _get_from_cache(self, key: str):
        """Retrieve from cache."""
        try:
            response = self.cache_client.get(
                f"{CACHE_SERVICE_URL}/cache/get",
                params={"key": key}
            )
            data = response.json()
            if data.get("hit"):
                return data.get("data")
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
        return None
    
    def _set_in_cache(self, key: str, data: dict, ttl: int):
        """Store in cache."""
        try:
            self.cache_client.post(
                f"{CACHE_SERVICE_URL}/cache/set",
                json={"key": key, "data": data, "ttl_seconds": ttl}
            )
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
    
    def _record_metrics(self, request, response, elapsed_ms: float):
        """Record metrics for model training."""
        try:
            user_type = getattr(request, '_user_type', 'guest')
            user_id = getattr(request, '_user_id', 'anonymous')
            cache_hit = response.get('X-Cache') == 'HIT'
            
            self.cache_client.post(
                f"{CACHE_SERVICE_URL}/metrics/record",
                json={
                    "endpoint": request.path,
                    "user_id": user_id,
                    "user_type": user_type,
                    "response_time_ms": elapsed_ms,
                    "cache_hit": cache_hit
                }
            )
        except Exception as e:
            logger.debug(f"Metrics error: {e}")

# settings.py configuration:
# Add to MIDDLEWARE:
MIDDLEWARE = [
    # ... other middleware ...
    'myapp.django_caching.IntelligentCacheMiddleware',
]
```

**Installation**:
```python
# settings.py
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    # ... other middleware ...
    'myapp.django_caching.IntelligentCacheMiddleware',  # Add this
]
```

---

## Async Python Integration

### For Async/Await Code

```python
# async_integration.py
"""
Async-first integration with Markov RL Cache
"""

import asyncio
import httpx
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

class AsyncCacheClient:
    """Async client for Markov RL Cache service."""
    
    def __init__(self, cache_service_url: str = "http://localhost:8000"):
        self.cache_service_url = cache_service_url
        self.client: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=30)
        return self
    
    async def __aexit__(self, *args):
        if self.client:
            await self.client.aclose()
    
    async def decide(self, endpoint: str, user_type: str, 
                    session_history: List[str] = None) -> Dict[str, Any]:
        """Get cache decision from intelligence system."""
        try:
            response = await self.client.post(
                f"{self.cache_service_url}/decide",
                json={
                    "endpoint": endpoint,
                    "user_type": user_type,
                    "session_history": session_history or []
                }
            )
            return response.json()
        except Exception as e:
            logger.error(f"Decision error: {e}")
            return {"action": "fetch_from_backend"}
    
    async def get(self, key: str) -> Optional[Dict]:
        """Get value from cache."""
        try:
            response = await self.client.get(
                f"{self.cache_service_url}/cache/get",
                params={"key": key}
            )
            data = response.json()
            return data.get("data") if data.get("hit") else None
        except Exception as e:
            logger.warning(f"Get error: {e}")
            return None
    
    async def set(self, key: str, value: Dict, ttl: int = 3600):
        """Set value in cache."""
        try:
            await self.client.post(
                f"{self.cache_service_url}/cache/set",
                json={"key": key, "data": value, "ttl_seconds": ttl}
            )
        except Exception as e:
            logger.warning(f"Set error: {e}")

# ===== USAGE PATTERNS =====

async def fetch_with_caching(
    endpoint: str,
    user_type: str,
    backend_fetch_fn,  # Async function to fetch from backend
    cache_service_url: str = "http://localhost:8000"
) -> Dict[str, Any]:
    """
    High-level function to fetch data with intelligent caching.
    
    Example:
        async def fetch_user(user_id):
            result = await httpx.get(f"http://api/users/{user_id}")
            return result.json()
        
        data = await fetch_with_caching(
            endpoint=f"users/{user_id}",
            user_type="premium",
            backend_fetch_fn=lambda: fetch_user(user_id)
        )
    """
    
    async with AsyncCacheClient(cache_service_url) as cache:
        # Step 1: Ask cache what to do
        decision = await cache.decide(endpoint, user_type)
        
        cache_key = f"api:{endpoint}"
        
        # Step 2: Try cache if recommended
        if decision.get("action") == "serve_from_cache":
            cached = await cache.get(cache_key)
            if cached:
                logger.info(f"Cache HIT: {endpoint}")
                return cached
        
        # Step 3: Fetch from backend
        logger.info(f"Cache MISS: {endpoint}")
        data = await backend_fetch_fn()
        
        # Step 4: Store in cache
        ttl = decision.get("ttl_seconds", 3600)
        await cache.set(cache_key, data, ttl)
        
        return data

# ===== EXAMPLE: BATCH PROCESSING =====

async def batch_fetch_with_caching(
    endpoints: List[Dict[str, str]],
    backend_fetch_fn,
    cache_service_url: str = "http://localhost:8000"
):
    """Fetch multiple endpoints with intelligent caching."""
    
    async with AsyncCacheClient(cache_service_url) as cache:
        tasks = []
        
        for endpoint_info in endpoints:
            task = fetch_with_caching(
                endpoint=endpoint_info['endpoint'],
                user_type=endpoint_info['user_type'],
                backend_fetch_fn=lambda: backend_fetch_fn(endpoint_info),
                cache_service_url=cache_service_url
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return [
            result if not isinstance(result, Exception) else None
            for result in results
        ]

# ===== EXAMPLE: STREAMING WITH CACHING =====

async def stream_with_caching(
    endpoint: str,
    user_type: str,
    backend_stream_fn,
    cache_service_url: str = "http://localhost:8000"
):
    """Stream data with caching support."""
    
    async with AsyncCacheClient(cache_service_url) as cache:
        decision = await cache.decide(endpoint, user_type)
        cache_key = f"api:{endpoint}"
        
        # Check cache first
        if decision.get("action") == "serve_from_cache":
            cached = await cache.get(cache_key)
            if cached:
                yield cached
                return
        
        # Stream from backend and cache
        buffer = []
        async for item in backend_stream_fn():
            yield item
            buffer.append(item)
        
        # Cache the complete result
        if buffer:
            await cache.set(cache_key, {"items": buffer})
```

---

## Go/Node.js Backend Communication

### Go HTTP Client

```go
// cache_client.go
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"time"
)

type CacheClient struct {
	baseURL    string
	httpClient *http.Client
}

type CacheDecision struct {
	Action       string   `json:"action"`
	TTLSeconds   int      `json:"ttl_seconds"`
	PrefetchList []string `json:"prefetch_list,omitempty"`
	Confidence   float64  `json:"confidence"`
}

func NewCacheClient(baseURL string) *CacheClient {
	return &CacheClient{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

func (c *CacheClient) GetDecision(ctx context.Context, endpoint, userType string) (*CacheDecision, error) {
	body := map[string]interface{}{
		"endpoint":   endpoint,
		"user_type":  userType,
	}
	
	bodyBytes, _ := json.Marshal(body)
	req, _ := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/decide", bytes.NewReader(bodyBytes))
	req.Header.Set("Content-Type", "application/json")
	
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	
	var decision CacheDecision
	json.NewDecoder(resp.Body).Decode(&decision)
	return &decision, nil
}

func (c *CacheClient) GetFromCache(ctx context.Context, key string) (interface{}, error) {
	req, _ := http.NewRequestWithContext(ctx, "GET", c.baseURL+"/cache/get?key="+key, nil)
	
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	
	var result map[string]interface{}
	json.NewDecoder(resp.Body).Decode(&result)
	
	if result["hit"].(bool) {
		return result["data"], nil
	}
	return nil, nil
}

// Usage
func ExampleUsage() {
	cache := NewCacheClient("http://cache-intelligence:8000")
	
	ctx := context.Background()
	decision, _ := cache.GetDecision(ctx, "products/123", "premium")
	
	if decision.Action == "serve_from_cache" {
		cached, _ := cache.GetFromCache(ctx, "api:products:123")
		// Use cached data
	}
}
```

### Node.js Axios Client

```javascript
// cache-client.js
const axios = require('axios');

class CacheClient {
  constructor(baseURL = 'http://cache-intelligence:8000') {
    this.baseURL = baseURL;
    this.client = axios.create({
      baseURL,
      timeout: 30000,
      headers: { 'Content-Type': 'application/json' }
    });
  }

  async getDecision(endpoint, userType) {
    try {
      const response = await this.client.post('/decide', {
        endpoint,
        user_type: userType
      });
      return response.data;
    } catch (error) {
      console.error('Cache decision error:', error.message);
      return { action: 'fetch_from_backend' };
    }
  }

  async getFromCache(key) {
    try {
      const response = await this.client.get(`/cache/get?key=${key}`);
      if (response.data.hit) {
        return response.data.data;
      }
    } catch (error) {
      console.warn('Cache get error:', error.message);
    }
    return null;
  }

  async setInCache(key, data, ttl = 3600) {
    try {
      await this.client.post('/cache/set', {
        key,
        data,
        ttl_seconds: ttl
      });
    } catch (error) {
      console.warn('Cache set error:', error.message);
    }
  }
}

// Express.js middleware
const expressMiddleware = (cacheClient) => {
  return async (req, res, next) => {
    if (req.method !== 'GET') {
      return next();
    }

    const userType = req.headers['x-user-type'] || 'guest';
    const decision = await cacheClient.getDecision(req.path, userType);

    if (decision.action === 'serve_from_cache') {
      const cached = await cacheClient.getFromCache(`api:${req.path}`);
      if (cached) {
        res.set('X-Cache', 'HIT');
        return res.json(cached);
      }
    }

    const originalJson = res.json;
    res.json = function(data) {
      const ttl = decision.ttl_seconds || 3600;
      cacheClient.setInCache(`api:${req.path}`, data, ttl);
      res.set('X-Cache', 'MISS');
      return originalJson.call(this, data);
    };

    next();
  };
};

module.exports = { CacheClient, expressMiddleware };
```

---

## Docker Compose Setup

### Complete Production Docker Compose

```yaml
# docker-compose.production.yml
version: '3.8'

networks:
  markov-network:
    driver: bridge

volumes:
  redis-data:
    driver: local
  prometheus-data:
    driver: local

services:
  # Cache Intelligence Service
  cache-intelligence:
    image: markov-rl-cache:latest
    container_name: cache-intel-prod
    ports:
      - "8000:8000"  # API
      - "9200:9200"  # Metrics
    environment:
      REDIS_HOST: redis
      REDIS_PORT: 6379
      REDIS_PASSWORD: ${REDIS_PASSWORD}
      API_WORKERS: 8
      LOG_LEVEL: INFO
      ENABLE_MONITORING: "true"
    depends_on:
      redis:
        condition: service_healthy
    networks:
      - markov-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    volumes:
      - ./models:/models:ro
      - ./logs:/app/logs

  # Redis Cache Backend
  redis:
    image: redis:7-alpine
    container_name: redis-prod
    command: >
      redis-server
      --maxmemory 2gb
      --maxmemory-policy allkeys-lru
      --appendonly yes
      --requirepass ${REDIS_PASSWORD}
    ports:
      - "6379:6379"
    networks:
      - markov-network
    volumes:
      - redis-data:/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 3G

  # Prometheus Monitoring
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus-prod
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=30d'
    networks:
      - markov-network
    restart: unless-stopped

  # Grafana Dashboards
  grafana:
    image: grafana/grafana:latest
    container_name: grafana-prod
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_PASSWORD}
      GF_INSTALL_PLUGINS: redis-datasource
    volumes:
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning:ro
    depends_on:
      - prometheus
    networks:
      - markov-network
    restart: unless-stopped
```

**Usage**:
```bash
# Create .env file
cat > .env <<EOF
REDIS_PASSWORD=your-secure-password
GRAFANA_PASSWORD=your-secure-password
EOF

# Start services
docker-compose -f docker-compose.production.yml up -d

# Verify
docker-compose ps
docker-compose logs -f cache-intelligence
```

---

## Kubernetes Manifests

### Complete K8s Deployment

See DEPLOYMENT_PLAYBOOK.md for full Kubernetes manifests.

---

## Monitoring & Metrics

### Prometheus Query Examples

```python
# Common queries for monitoring

queries = {
    "cache_hit_rate": '''
        rate(markov_rl_cache_hits_total[5m]) / 
        (rate(markov_rl_cache_hits_total[5m]) + 
         rate(markov_rl_cache_misses_total[5m]))
    ''',
    
    "avg_latency": '''
        rate(markov_rl_api_request_duration_ms_sum[5m]) /
        rate(markov_rl_api_request_duration_ms_count[5m])
    ''',
    
    "error_rate": '''
        rate(markov_rl_api_errors_total[5m])
    ''',
    
    "prediction_accuracy": '''
        markov_rl_predictions_correct_at_k{k="1"}
    ''',
    
    "agent_reward": '''
        markov_rl_episode_reward
    ''',
    
    "cascade_risk": '''
        markov_rl_cascade_risk_score
    '''
}
```

### Grafana Dashboard JSON

```json
{
  "dashboard": {
    "title": "Markov RL Cache Intelligence",
    "panels": [
      {
        "title": "Cache Hit Rate (%)",
        "targets": [{
          "expr": "rate(markov_rl_cache_hits_total[5m]) / (rate(markov_rl_cache_hits_total[5m]) + rate(markov_rl_cache_misses_total[5m])) * 100"
        }],
        "type": "gauge"
      },
      {
        "title": "Average Response Latency (ms)",
        "targets": [{
          "expr": "rate(markov_rl_api_request_duration_ms_sum[5m]) / rate(markov_rl_api_request_duration_ms_count[5m])"
        }],
        "type": "graph"
      },
      {
        "title": "Prediction Accuracy (Top-1)",
        "targets": [{
          "expr": "markov_rl_predictions_correct_at_k{k=\"1\"}"
        }],
        "type": "graph"
      }
    ]
  }
}
```

---

**End of Implementation Reference**


