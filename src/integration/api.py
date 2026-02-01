"""
FastAPI control API for the IntegrationController.

Provides REST endpoints for remote management and monitoring
of the intelligent caching system.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logging.warning("FastAPI not installed. API functionality disabled.")

logger = logging.getLogger(__name__)


# Request/Response models
if FASTAPI_AVAILABLE:
    class TrainRequest(BaseModel):
        """Request to start training."""
        num_episodes: Optional[int] = None
        
    class EvaluateRequest(BaseModel):
        """Request to run evaluation."""
        num_episodes: int = 10
        
    class ActionRequest(BaseModel):
        """Request to get action for state."""
        state: list
        
    class FailureRequest(BaseModel):
        """Request to inject failure."""
        failure_type: str  # 'latency', 'cascade', 'timeout'
        severity: float = 0.5
        
    class ConfigUpdateRequest(BaseModel):
        """Request to update configuration."""
        config: Dict[str, Any]
        
    class APICallRequest(BaseModel):
        """Request to process an API call."""
        endpoint: str
        context: Optional[Dict[str, Any]] = None


def create_app(controller) -> FastAPI:
    """
    Create FastAPI app with controller integration.
    
    Args:
        controller: IntegrationController instance
    
    Returns:
        FastAPI application
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI not installed. Install with: pip install fastapi uvicorn")
    
    app = FastAPI(
        title="Intelligent Caching System API",
        description="Control API for the RL-based intelligent caching system",
        version="1.0.0"
    )
    
    # Store controller reference
    app.state.controller = controller
    
    # Background training task
    app.state.training_task = None
    
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "service": "Intelligent Caching System",
            "version": "1.0.0",
            "status": "running",
            "mode": controller.config.mode
        }
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        status = controller.get_status()
        
        # Determine health status
        all_healthy = all(status['component_health'].values())
        
        return {
            "status": "healthy" if all_healthy else "degraded",
            "timestamp": datetime.now().isoformat(),
            "is_setup": status['is_setup'],
            "is_running": status['is_running'],
            "components": status['component_health']
        }
    
    @app.get("/status")
    async def get_status():
        """Get current system status."""
        try:
            status = controller.get_status()
            return JSONResponse(content=status)
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/metrics")
    async def get_metrics():
        """Get current metrics from all components."""
        try:
            metrics = controller.get_metrics()
            
            # Also expose Prometheus metrics if enabled
            if controller.config.enable_monitoring and controller.metrics_registry:
                from prometheus_client import generate_latest
                prometheus_metrics = generate_latest(controller.metrics_registry).decode('utf-8')
                metrics['prometheus'] = prometheus_metrics
            
            return JSONResponse(content=metrics)
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/train/start")
    async def start_training(request: TrainRequest, background_tasks: BackgroundTasks):
        """Start training in background."""
        try:
            if controller.config.mode != "training":
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot train in {controller.config.mode} mode"
                )
            
            if app.state.training_task is not None:
                raise HTTPException(status_code=400, detail="Training already in progress")
            
            # Start training in background
            def train_task():
                try:
                    logger.info("Starting background training...")
                    result = controller.train(num_episodes=request.num_episodes)
                    logger.info(f"Training complete: {result}")
                    app.state.training_task = None
                except Exception as e:
                    logger.error(f"Training failed: {e}")
                    app.state.training_task = None
            
            background_tasks.add_task(train_task)
            app.state.training_task = "running"
            
            return {
                "message": "Training started",
                "num_episodes": request.num_episodes or controller.config.training_config.num_episodes
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error starting training: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/train/stop")
    async def stop_training():
        """Stop ongoing training."""
        try:
            controller._training_interrupted = True
            app.state.training_task = None
            
            return {"message": "Training stop signal sent"}
            
        except Exception as e:
            logger.error(f"Error stopping training: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/train/progress")
    async def get_training_progress():
        """Get current training progress."""
        try:
            if controller.config.mode != "training":
                raise HTTPException(
                    status_code=400,
                    detail=f"Not in training mode (current: {controller.config.mode})"
                )
            
            status = controller.get_status()
            progress = status.get('training_progress', {})
            
            progress['is_training'] = app.state.training_task is not None
            
            return JSONResponse(content=progress)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting training progress: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/evaluate")
    async def run_evaluation(request: EvaluateRequest):
        """Run evaluation episodes."""
        try:
            logger.info(f"Running evaluation for {request.num_episodes} episodes...")
            results = controller.evaluate(num_episodes=request.num_episodes)
            
            return JSONResponse(content=results)
            
        except Exception as e:
            logger.error(f"Error running evaluation: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/action")
    async def get_action(request: ActionRequest):
        """Get recommended action for given state."""
        try:
            import numpy as np
            
            state = np.array(request.state, dtype=np.float32)
            action = controller.predict_action(state)
            
            return {
                "action": int(action),
                "action_name": controller._get_action_name(action),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting action: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api-call")
    async def process_api_call(request: APICallRequest):
        """Process an API call through the system."""
        try:
            result = controller.process_api_call(
                endpoint=request.endpoint,
                context=request.context
            )
            
            return JSONResponse(content=result)
            
        except Exception as e:
            logger.error(f"Error processing API call: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/inject/failure")
    async def inject_failure(request: FailureRequest):
        """Inject a failure scenario for testing."""
        try:
            if not controller.env:
                raise HTTPException(status_code=400, detail="Environment not available")
            
            # Inject failure based on type
            if request.failure_type == 'latency':
                # Increase system latency
                controller.env.system_metrics['p99_latency'] *= (1 + request.severity)
                message = f"Injected latency spike (severity={request.severity})"
                
            elif request.failure_type == 'cascade':
                # Trigger cascade conditions
                controller.env.system_metrics['cpu'] = 0.9
                controller.env.system_metrics['error_rate'] = 0.2
                message = f"Injected cascade conditions (severity={request.severity})"
                
            elif request.failure_type == 'timeout':
                # Simulate timeouts
                controller.env.system_metrics['p95_latency'] *= 2
                message = f"Injected timeout scenario (severity={request.severity})"
                
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown failure type: {request.failure_type}"
                )
            
            logger.info(message)
            
            return {
                "message": message,
                "failure_type": request.failure_type,
                "severity": request.severity
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error injecting failure: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/inject/restore")
    async def restore_from_failure():
        """Restore system from injected failures."""
        try:
            if not controller.env:
                raise HTTPException(status_code=400, detail="Environment not available")
            
            # Reset to normal metrics
            controller.env.system_metrics = controller.env._initialize_system_metrics()
            
            return {
                "message": "System restored to normal state",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error restoring system: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/cache/contents")
    async def get_cache_contents():
        """Get current cache contents."""
        try:
            if not controller.cache_manager:
                raise HTTPException(status_code=400, detail="Cache manager not available")
            
            # Get cache metrics (contents would require backend support)
            metrics = controller.cache_manager.get_metrics()
            
            return {
                "metrics": metrics,
                "message": "Full cache contents listing requires backend support",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting cache contents: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.delete("/cache/clear")
    async def clear_cache():
        """Clear the cache."""
        try:
            if not controller.cache_manager:
                raise HTTPException(status_code=400, detail="Cache manager not available")
            
            # Stop and restart cache manager to clear
            controller.cache_manager.stop()
            controller.cache_manager.start()
            
            return {
                "message": "Cache cleared",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/config/update")
    async def update_config(request: ConfigUpdateRequest):
        """Update configuration (limited support)."""
        try:
            # Only allow updating certain config values
            allowed_keys = ['log_level', 'enable_monitoring']
            
            updated = {}
            for key, value in request.config.items():
                if key in allowed_keys:
                    setattr(controller.config, key, value)
                    updated[key] = value
                    
                    # Apply changes
                    if key == 'log_level':
                        log_level = getattr(logging, value.upper())
                        logging.getLogger().setLevel(log_level)
                        logger.setLevel(log_level)
            
            return {
                "message": "Configuration updated",
                "updated": updated,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error updating config: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


def serve_api(controller, host: str = "0.0.0.0", port: int = 8080):
    """
    Start the API server.
    
    Args:
        controller: IntegrationController instance
        host: Host to bind to
        port: Port to listen on
    """
    if not FASTAPI_AVAILABLE:
        logger.error("FastAPI not installed. Cannot start API server.")
        return
    
    app = create_app(controller)
    
    logger.info(f"Starting API server on {host}:{port}")
    
    uvicorn.run(app, host=host, port=port)

