"""
Traffic Generator - Realistic user behavior simulation

Simulates realistic user traffic patterns including:
- Different user types (premium, free, guest)
- Realistic workflows (browse, purchase, account management)
- Time-based traffic patterns
- Gradual ramp-up/down
- Back-pressure handling
"""

import asyncio
import random
import time
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from collections import defaultdict, deque
import aiohttp
import logging


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TrafficProfile:
    """Configuration for traffic generation patterns.

    Attributes:
        name: Profile identifier
        requests_per_second: Target request rate
        duration_seconds: How long to run
        user_type_distribution: Mapping of user types to fractions
        workflow_distribution: Mapping of workflow names to fractions
        ramp_up_seconds: Time to gradually reach target rate
        ramp_down_seconds: Time to gradually decrease at end
    """
    name: str
    requests_per_second: float
    duration_seconds: int
    user_type_distribution: Dict[str, float] = field(default_factory=dict)
    workflow_distribution: Dict[str, float] = field(default_factory=dict)
    ramp_up_seconds: int = 0
    ramp_down_seconds: int = 0

    @classmethod
    def from_yaml(cls, filepath: str) -> 'TrafficProfile':
        """Load traffic profile from YAML file.

        Args:
            filepath: Path to YAML file

        Returns:
            TrafficProfile instance
        """
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)

        return cls(
            name=data.get('name', 'unnamed'),
            requests_per_second=data.get('requests_per_second', 100),
            duration_seconds=data.get('duration_seconds', 60),
            user_type_distribution=data.get('user_type_distribution', {}),
            workflow_distribution=data.get('workflow_distribution', {}),
            ramp_up_seconds=data.get('ramp_up_seconds', 0),
            ramp_down_seconds=data.get('ramp_down_seconds', 0)
        )


# ============================================================================
# User Workflows
# ============================================================================

class BaseWorkflow:
    """Base class for user workflows."""

    def __init__(self, user_id: str, user_type: str, service_urls: Dict[str, str]):
        """Initialize workflow.

        Args:
            user_id: User identifier
            user_type: Type of user (premium, free, guest)
            service_urls: Mapping of service names to base URLs
        """
        self.user_id = user_id
        self.user_type = user_type
        self.service_urls = service_urls
        self.current_step = 0
        self.auth_token: Optional[str] = None
        self.context: Dict[str, Any] = {}
        self.steps = self._define_steps()

    def _define_steps(self) -> List[Tuple[str, str, Dict[str, Any], callable]]:
        """Define workflow steps.

        Returns:
            List of (service, endpoint, params, next_step_decider) tuples
        """
        raise NotImplementedError

    def get_next_request(self) -> Optional[Tuple[str, str, Dict[str, Any]]]:
        """Get the next API call to make.

        Returns:
            Tuple of (full_url, method, params) or None if workflow complete
        """
        if self.is_complete():
            return None

        service, endpoint, params, _ = self.steps[self.current_step]

        # Build full URL
        base_url = self.service_urls.get(service, f"http://localhost:8000")
        full_url = f"{base_url}{endpoint}"

        # Replace placeholders in URL and params
        full_url = self._substitute_context(full_url)
        params = self._substitute_context_dict(params)

        # Add auth token if available
        if self.auth_token and 'headers' in params:
            params['headers']['Authorization'] = f"Bearer {self.auth_token}"

        return full_url, params.get('method', 'GET'), params

    def advance(self, response: Optional[Dict[str, Any]] = None):
        """Move to next step in workflow.

        Args:
            response: Response from previous request
        """
        if response:
            self._process_response(response)

        if self.current_step < len(self.steps):
            _, _, _, next_step_decider = self.steps[self.current_step]
            next_step = next_step_decider(self, response)
            self.current_step = next_step
        else:
            self.current_step = len(self.steps)

    def is_complete(self) -> bool:
        """Check if workflow is finished."""
        return self.current_step >= len(self.steps)

    def _process_response(self, response: Dict[str, Any]):
        """Process response and update context."""
        # Extract useful data from response
        if isinstance(response, dict):
            if 'token' in response:
                self.auth_token = response['token']
            if 'user_id' in response:
                self.context['user_id'] = response['user_id']
            if 'product_id' in response:
                self.context['product_id'] = response['product_id']
            if 'order_id' in response:
                self.context['order_id'] = response['order_id']

    def _substitute_context(self, text: str) -> str:
        """Replace {variable} placeholders with context values."""
        for key, value in self.context.items():
            text = text.replace(f"{{{key}}}", str(value))
        return text

    def _substitute_context_dict(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively substitute context in params dict."""
        result = {}
        for key, value in params.items():
            if isinstance(value, str):
                result[key] = self._substitute_context(value)
            elif isinstance(value, dict):
                result[key] = self._substitute_context_dict(value)
            else:
                result[key] = value
        return result


class BrowseWorkflow(BaseWorkflow):
    """User browsing products without necessarily buying.

    Entry: /products or /search
    Flow: products → product details → (maybe back to products) → (maybe add to cart)
    Duration: 2-10 API calls
    """

    def _define_steps(self) -> List[Tuple[str, str, Dict[str, Any], callable]]:
        """Define browsing workflow steps."""
        steps = []

        # Step 0: Browse products or search
        if random.random() < 0.5:
            steps.append((
                'product',
                '/products',
                {'method': 'GET', 'params': {'page': 1, 'page_size': 20}},
                lambda w, r: 1  # Always go to next step
            ))
        else:
            steps.append((
                'product',
                '/search',
                {'method': 'GET', 'params': {'q': random.choice(['laptop', 'phone', 'book', 'shirt'])}},
                lambda w, r: 1
            ))

        # Step 1: View product details
        steps.append((
            'product',
            f'/products/prod_{random.randint(0, 99):03d}',
            {'method': 'GET'},
            lambda w, r: 2 if random.random() < 0.7 else len(w.steps)  # 70% continue
        ))

        # Step 2: Maybe view another product
        if random.random() < 0.6:
            steps.append((
                'product',
                f'/products/prod_{random.randint(0, 99):03d}',
                {'method': 'GET'},
                lambda w, r: 3 if random.random() < 0.4 else len(w.steps)  # 40% add to cart
            ))

        # Step 3: Maybe add to cart (if logged in)
        if random.random() < 0.3:
            steps.append((
                'cart',
                '/cart/add',
                {
                    'method': 'POST',
                    'json': {
                        'user_id': self.user_id,
                        'product_id': f'prod_{random.randint(0, 99):03d}',
                        'quantity': 1
                    }
                },
                lambda w, r: len(w.steps)  # Done after adding to cart
            ))

        return steps


class PurchaseWorkflow(BaseWorkflow):
    """Complete purchase workflow.

    Entry: /login
    Flow: login → profile → browse → product → add to cart → view cart → checkout → payment
    Duration: 5-12 API calls
    Higher completion rate for premium users
    """

    def _define_steps(self) -> List[Tuple[str, str, Dict[str, Any], callable]]:
        """Define purchase workflow steps."""
        completion_rate = 0.8 if self.user_type == "premium" else 0.5

        steps = [
            # Step 0: Login
            (
                'user',
                '/login',
                {
                    'method': 'POST',
                    'json': {'username': 'test', 'password': 'test'}
                },
                lambda w, r: 1 if r and r.get('status') == 'success' else len(w.steps)
            ),

            # Step 1: Get profile
            (
                'user',
                '/profile',
                {'method': 'GET', 'params': {'user_id': self.user_id}},
                lambda w, r: 2
            ),

            # Step 2: Browse products
            (
                'product',
                '/products',
                {'method': 'GET', 'params': {'page': 1}},
                lambda w, r: 3
            ),

            # Step 3: View product details
            (
                'product',
                f'/products/prod_{random.randint(0, 99):03d}',
                {'method': 'GET'},
                lambda w, r: 4 if random.random() < completion_rate else len(w.steps)
            ),

            # Step 4: Add to cart
            (
                'cart',
                '/cart/add',
                {
                    'method': 'POST',
                    'json': {
                        'user_id': self.user_id,
                        'product_id': f'prod_{random.randint(0, 99):03d}',
                        'quantity': random.randint(1, 3)
                    }
                },
                lambda w, r: 5 if random.random() < completion_rate else len(w.steps)
            ),

            # Step 5: View cart
            (
                'cart',
                '/cart',
                {'method': 'GET', 'params': {'user_id': self.user_id}},
                lambda w, r: 6 if random.random() < completion_rate else len(w.steps)
            ),

            # Step 6: Create order
            (
                'order',
                '/orders',
                {
                    'method': 'POST',
                    'json': {
                        'user_id': self.user_id,
                        'payment_method': {'type': 'credit_card', 'card_number': '4532********1234'},
                        'shipping_address': '123 Main St, City, State 12345'
                    }
                },
                lambda w, r: len(w.steps)  # Done after order
            ),
        ]

        return steps


class AccountWorkflow(BaseWorkflow):
    """Account management workflow.

    Entry: /login
    Flow: login → profile → orders → order details
    Duration: 3-6 API calls
    """

    def _define_steps(self) -> List[Tuple[str, str, Dict[str, Any], callable]]:
        """Define account workflow steps."""
        steps = [
            # Step 0: Login
            (
                'user',
                '/login',
                {
                    'method': 'POST',
                    'json': {'username': 'test', 'password': 'test'}
                },
                lambda w, r: 1 if r and r.get('status') == 'success' else len(w.steps)
            ),

            # Step 1: Get profile
            (
                'user',
                '/profile',
                {'method': 'GET', 'params': {'user_id': self.user_id}},
                lambda w, r: 2
            ),

            # Step 2: List orders
            (
                'order',
                '/orders',
                {'method': 'GET', 'params': {'user_id': self.user_id}},
                lambda w, r: 3 if r and r.get('count', 0) > 0 else len(w.steps)
            ),

            # Step 3: View order details (if orders exist)
            (
                'order',
                '/orders/{order_id}',
                {'method': 'GET'},
                lambda w, r: len(w.steps)
            ),
        ]

        return steps


class QuickBuyWorkflow(BaseWorkflow):
    """Quick purchase workflow for premium users.

    Entry: /login
    Flow: login → search → product → add to cart → checkout → payment
    Duration: 6-8 calls
    Faster transitions (premium users)
    """

    def _define_steps(self) -> List[Tuple[str, str, Dict[str, Any], callable]]:
        """Define quick buy workflow steps."""
        steps = [
            # Step 0: Login
            (
                'user',
                '/login',
                {
                    'method': 'POST',
                    'json': {'username': 'test', 'password': 'test'}
                },
                lambda w, r: 1 if r and r.get('status') == 'success' else len(w.steps)
            ),

            # Step 1: Search for specific item
            (
                'product',
                '/search',
                {'method': 'GET', 'params': {'q': random.choice(['laptop', 'phone', 'book'])}},
                lambda w, r: 2
            ),

            # Step 2: View product
            (
                'product',
                f'/products/prod_{random.randint(0, 99):03d}',
                {'method': 'GET'},
                lambda w, r: 3
            ),

            # Step 3: Add to cart
            (
                'cart',
                '/cart/add',
                {
                    'method': 'POST',
                    'json': {
                        'user_id': self.user_id,
                        'product_id': f'prod_{random.randint(0, 99):03d}',
                        'quantity': 1
                    }
                },
                lambda w, r: 4
            ),

            # Step 4: Create order (skip viewing cart)
            (
                'order',
                '/orders',
                {
                    'method': 'POST',
                    'json': {
                        'user_id': self.user_id,
                        'payment_method': {'type': 'credit_card'},
                        'shipping_address': '123 Main St'
                    }
                },
                lambda w, r: len(w.steps)
            ),
        ]

        return steps


# Workflow registry
WORKFLOW_CLASSES = {
    'browse': BrowseWorkflow,
    'purchase': PurchaseWorkflow,
    'account': AccountWorkflow,
    'quickbuy': QuickBuyWorkflow,
}


# ============================================================================
# Simulated User
# ============================================================================

class SimulatedUser:
    """Represents one user going through a workflow."""

    def __init__(
        self,
        user_id: str,
        user_type: str,
        workflow_name: str,
        service_urls: Dict[str, str]
    ):
        """Initialize simulated user.

        Args:
            user_id: User identifier
            user_type: Type of user (premium, free, guest)
            workflow_name: Name of workflow to execute
            service_urls: Mapping of service names to base URLs
        """
        self.user_id = user_id
        self.user_type = user_type
        self.workflow_name = workflow_name
        self.created_at = time.time()

        # Create workflow instance
        workflow_class = WORKFLOW_CLASSES.get(workflow_name, BrowseWorkflow)
        self.workflow = workflow_class(user_id, user_type, service_urls)

    def get_next_request(self) -> Optional[Tuple[str, str, Dict[str, Any]]]:
        """Get the next API call to make."""
        return self.workflow.get_next_request()

    def advance(self, response: Optional[Dict[str, Any]] = None):
        """Move to next step in workflow."""
        self.workflow.advance(response)

    def is_complete(self) -> bool:
        """Check if workflow is finished."""
        return self.workflow.is_complete()


# ============================================================================
# Traffic Generator
# ============================================================================

class TrafficGenerator:
    """Generates realistic traffic to e-commerce services."""

    def __init__(self, profile: TrafficProfile, service_urls: Dict[str, str]):
        """Initialize traffic generator.

        Args:
            profile: Traffic profile configuration
            service_urls: Mapping of service names to base URLs
        """
        self.profile = profile
        self.service_urls = service_urls

        self.logger = logging.getLogger(f"TrafficGenerator.{profile.name}")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # State
        self.running = False
        self.paused = False
        self.active_users: List[SimulatedUser] = []
        self.user_counter = 0

        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'latencies': deque(maxlen=10000),
            'requests_by_workflow': defaultdict(int),
            'start_time': None,
            'end_time': None,
        }

        # HTTP session
        self.session: Optional[aiohttp.ClientSession] = None

        # Tasks
        self.generator_task: Optional[asyncio.Task] = None
        self.executor_task: Optional[asyncio.Task] = None
        self.request_queue: asyncio.Queue = asyncio.Queue()

    async def start(self):
        """Begin generating traffic asynchronously."""
        if self.running:
            self.logger.warning("Traffic generator already running")
            return

        self.running = True
        self.stats['start_time'] = time.time()

        # Create HTTP session
        self.session = aiohttp.ClientSession()

        # Start generator and executor tasks
        self.generator_task = asyncio.create_task(self._user_generator())
        self.executor_task = asyncio.create_task(self._request_executor())

        self.logger.info(
            f"Traffic generator started: {self.profile.name}, "
            f"{self.profile.requests_per_second} RPS for {self.profile.duration_seconds}s"
        )

    async def stop(self):
        """Gracefully stop traffic generation."""
        if not self.running:
            return

        self.logger.info("Stopping traffic generator...")
        self.running = False

        # Cancel tasks
        if self.generator_task:
            self.generator_task.cancel()
            try:
                await self.generator_task
            except asyncio.CancelledError:
                pass

        if self.executor_task:
            self.executor_task.cancel()
            try:
                await self.executor_task
            except asyncio.CancelledError:
                pass

        # Close HTTP session
        if self.session:
            await self.session.close()

        self.stats['end_time'] = time.time()
        self.logger.info("Traffic generator stopped")

    def pause(self):
        """Pause traffic generation."""
        self.paused = True
        self.logger.info("Traffic generator paused")

    def resume(self):
        """Resume traffic generation."""
        self.paused = False
        self.logger.info("Traffic generator resumed")

    async def _user_generator(self):
        """Generate simulated users at configured rate."""
        start_time = time.time()
        duration = self.profile.duration_seconds
        ramp_up = self.profile.ramp_up_seconds
        ramp_down = self.profile.ramp_down_seconds
        target_rps = self.profile.requests_per_second

        try:
            while self.running:
                if self.paused:
                    await asyncio.sleep(0.1)
                    continue

                elapsed = time.time() - start_time

                # Check if we've exceeded duration
                if elapsed > duration:
                    break

                # Calculate current RPS based on ramp-up/down
                if elapsed < ramp_up:
                    # Ramp up
                    current_rps = target_rps * (elapsed / ramp_up)
                elif elapsed > duration - ramp_down:
                    # Ramp down
                    remaining = duration - elapsed
                    current_rps = target_rps * (remaining / ramp_down)
                else:
                    # Normal rate
                    current_rps = target_rps

                # Create new user
                if current_rps > 0:
                    user = self._create_user()
                    self.active_users.append(user)

                    # Queue first request
                    await self._queue_user_request(user)

                    # Sleep to maintain rate
                    sleep_time = 1.0 / max(current_rps, 0.1)
                    await asyncio.sleep(sleep_time)
                else:
                    await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Error in user generator: {e}")

    async def _request_executor(self):
        """Execute requests from the queue."""
        try:
            while self.running or not self.request_queue.empty():
                if self.paused:
                    await asyncio.sleep(0.1)
                    continue

                try:
                    # Get request from queue with timeout
                    user, request_data = await asyncio.wait_for(
                        self.request_queue.get(),
                        timeout=1.0
                    )

                    # Execute request
                    await self._execute_request(user, request_data)

                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    self.logger.error(f"Error in request executor: {e}")

        except asyncio.CancelledError:
            pass

    async def _queue_user_request(self, user: SimulatedUser):
        """Queue the next request for a user."""
        request_data = user.get_next_request()
        if request_data:
            await self.request_queue.put((user, request_data))

    async def _execute_request(
        self,
        user: SimulatedUser,
        request_data: Tuple[str, str, Dict[str, Any]]
    ):
        """Execute a single request.

        Args:
            user: SimulatedUser making the request
            request_data: Tuple of (url, method, params)
        """
        url, method, params = request_data

        start_time = time.time()
        response_data = None
        success = False

        try:
            # Make HTTP request
            async with self.session.request(
                method,
                url,
                json=params.get('json'),
                params=params.get('params'),
                headers=params.get('headers', {}),
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                try:
                    response_data = await response.json()
                except:
                    response_data = {}

                success = response.status < 400

        except asyncio.TimeoutError:
            self.logger.debug(f"Request timeout: {url}")
        except aiohttp.ClientError as e:
            self.logger.debug(f"Request error: {url} - {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error: {url} - {e}")

        # Record statistics
        latency_ms = (time.time() - start_time) * 1000
        self.stats['total_requests'] += 1
        if success:
            self.stats['successful_requests'] += 1
        else:
            self.stats['failed_requests'] += 1

        self.stats['latencies'].append(latency_ms)
        self.stats['requests_by_workflow'][user.workflow_name] += 1

        # Advance user workflow
        user.advance(response_data)

        # Queue next request if workflow not complete
        if not user.is_complete():
            # Add realistic delay between requests (think time)
            await asyncio.sleep(random.uniform(0.5, 2.0))
            await self._queue_user_request(user)
        else:
            # Remove completed user
            if user in self.active_users:
                self.active_users.remove(user)

    def _create_user(self) -> SimulatedUser:
        """Create a new simulated user."""
        self.user_counter += 1
        user_id = f"simuser_{self.user_counter:06d}"

        # Select user type based on distribution
        user_type = self._weighted_choice(
            self.profile.user_type_distribution,
            default='free'
        )

        # Select workflow based on distribution
        workflow_name = self._weighted_choice(
            self.profile.workflow_distribution,
            default='browse'
        )

        return SimulatedUser(user_id, user_type, workflow_name, self.service_urls)

    def _weighted_choice(self, distribution: Dict[str, float], default: str) -> str:
        """Choose an option based on weighted distribution."""
        if not distribution:
            return default

        choices = list(distribution.keys())
        weights = list(distribution.values())

        # Normalize weights
        total = sum(weights)
        if total == 0:
            return default

        weights = [w / total for w in weights]

        return random.choices(choices, weights=weights)[0]

    def get_stats(self) -> Dict[str, Any]:
        """Return current traffic statistics."""
        latencies = list(self.stats['latencies'])

        if latencies:
            sorted_latencies = sorted(latencies)
            n = len(sorted_latencies)
            latency_p50 = sorted_latencies[n // 2]
            latency_p95 = sorted_latencies[int(n * 0.95)]
            latency_p99 = sorted_latencies[int(n * 0.99)]
            latency_mean = sum(latencies) / n
        else:
            latency_p50 = latency_p95 = latency_p99 = latency_mean = 0

        total = self.stats['total_requests']
        success_rate = self.stats['successful_requests'] / total if total > 0 else 0

        # Calculate throughput
        if self.stats['start_time']:
            if self.stats['end_time']:
                duration = self.stats['end_time'] - self.stats['start_time']
            else:
                duration = time.time() - self.stats['start_time']

            throughput = total / duration if duration > 0 else 0
        else:
            throughput = 0

        return {
            'total': total,
            'successful': self.stats['successful_requests'],
            'failed': self.stats['failed_requests'],
            'success_rate': success_rate,
            'latency_mean': latency_mean,
            'latency_p50': latency_p50,
            'latency_p95': latency_p95,
            'latency_p99': latency_p99,
            'throughput_rps': throughput,
            'active_users': len(self.active_users),
            'requests_by_workflow': dict(self.stats['requests_by_workflow']),
        }

    def reset_stats(self):
        """Clear statistics."""
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'latencies': deque(maxlen=10000),
            'requests_by_workflow': defaultdict(int),
            'start_time': None,
            'end_time': None,
        }
        self.logger.info("Statistics reset")

