"""Synthetic data generator for creating realistic API call traces with known patterns.

This module generates synthetic but realistic API trace data where we control and know
the true underlying patterns. This is crucial for validating that our Markov chain and
RL models learn the correct transitions and behaviors.
"""

import random
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Tuple
import yaml
from pathlib import Path

from preprocessing.models import APICall, Session, Dataset


@dataclass
class WorkflowDefinition:
    """Describes a user behavior pattern with transition probabilities.

    This defines a Markov chain of user behavior that we can use to generate
    synthetic data with known ground truth patterns.

    Attributes:
        name: Identifier for this workflow
        entry_points: Starting endpoints with probabilities (must sum to 1.0)
        transitions: Nested dict of transition probabilities
                     transitions[from_endpoint][to_endpoint] = probability
        exit_points: Endpoints that typically end a session
        avg_response_times: Optional dict of endpoint → typical response time (ms)
        user_type_distribution: Distribution of user types {type: probability}
    """
    name: str
    entry_points: Dict[str, float]
    transitions: Dict[str, Dict[str, float]]
    exit_points: Set[str]
    avg_response_times: Dict[str, float] = field(default_factory=dict)
    user_type_distribution: Dict[str, float] = field(default_factory=lambda: {
        'premium': 0.3,
        'free': 0.5,
        'guest': 0.2
    })

    def __post_init__(self):
        """Validate the workflow definition."""
        # Validate entry points sum to 1.0
        entry_sum = sum(self.entry_points.values())
        if not (0.99 <= entry_sum <= 1.01):
            raise ValueError(f"Entry points must sum to 1.0, got {entry_sum}")

        # Validate transitions sum to 1.0 for each endpoint
        for endpoint, next_endpoints in self.transitions.items():
            trans_sum = sum(next_endpoints.values())
            if not (0.99 <= trans_sum <= 1.01):
                raise ValueError(f"Transitions from {endpoint} must sum to 1.0, got {trans_sum}")

        # Validate user type distribution
        user_sum = sum(self.user_type_distribution.values())
        if not (0.99 <= user_sum <= 1.01):
            raise ValueError(f"User type distribution must sum to 1.0, got {user_sum}")

    def to_yaml(self, filepath: Path) -> None:
        """Save workflow definition to YAML file."""
        data = {
            'name': self.name,
            'entry_points': self.entry_points,
            'transitions': self.transitions,
            'exit_points': list(self.exit_points),
            'avg_response_times': self.avg_response_times,
            'user_type_distribution': self.user_type_distribution
        }

        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

    @classmethod
    def from_yaml(cls, filepath: Path) -> 'WorkflowDefinition':
        """Load workflow definition from YAML file."""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)

        return cls(
            name=data['name'],
            entry_points=data['entry_points'],
            transitions=data['transitions'],
            exit_points=set(data['exit_points']),
            avg_response_times=data.get('avg_response_times', {}),
            user_type_distribution=data.get('user_type_distribution', {
                'premium': 0.3,
                'free': 0.5,
                'guest': 0.2
            })
        )


class SyntheticGenerator:
    """Generates synthetic API call traces with realistic patterns.

    This generator creates data where we know the true underlying Markov chain,
    allowing us to validate that our models learn the correct patterns.
    """

    # Pre-built e-commerce workflow with realistic patterns
    ECOMMERCE_WORKFLOW = WorkflowDefinition(
        name="ecommerce",
        entry_points={
            "/api/login": 0.60,          # Most users start with login
            "/api/products/browse": 0.30,  # Some go straight to browsing
            "/api/products/search": 0.10   # Some start with search
        },
        transitions={
            # After login
            "/api/login": {
                "/api/users/{id}/profile": 0.85,
                "/api/products/browse": 0.15
            },
            # From profile
            "/api/users/{id}/profile": {
                "/api/products/browse": 0.50,
                "/api/users/{id}/orders": 0.30,
                "/api/users/{id}/settings": 0.15,
                "/api/logout": 0.05
            },
            # From browsing
            "/api/products/browse": {
                "/api/products/{id}/details": 0.60,
                "/api/products/search": 0.25,
                "/api/cart": 0.10,
                "/api/logout": 0.05
            },
            # From product details
            "/api/products/{id}/details": {
                "/api/cart/add": 0.40,
                "/api/products/browse": 0.35,
                "/api/products/search": 0.15,
                "/api/products/{id}/reviews": 0.10
            },
            # From product reviews
            "/api/products/{id}/reviews": {
                "/api/cart/add": 0.50,
                "/api/products/browse": 0.30,
                "/api/products/{id}/details": 0.20
            },
            # From search
            "/api/products/search": {
                "/api/products/{id}/details": 0.70,
                "/api/products/browse": 0.20,
                "/api/logout": 0.10
            },
            # From cart
            "/api/cart": {
                "/api/checkout": 0.50,
                "/api/products/browse": 0.30,
                "/api/cart/remove": 0.15,
                "/api/logout": 0.05
            },
            # From cart add
            "/api/cart/add": {
                "/api/cart": 0.70,
                "/api/products/browse": 0.20,
                "/api/checkout": 0.10
            },
            # From cart remove
            "/api/cart/remove": {
                "/api/cart": 0.50,
                "/api/products/browse": 0.40,
                "/api/logout": 0.10
            },
            # From checkout
            "/api/checkout": {
                "/api/payment": 0.90,
                "/api/cart": 0.10
            },
            # From payment
            "/api/payment": {
                "/api/confirmation": 0.95,
                "/api/cart": 0.05
            },
            # From orders
            "/api/users/{id}/orders": {
                "/api/orders/{id}/details": 0.60,
                "/api/products/browse": 0.25,
                "/api/users/{id}/profile": 0.10,
                "/api/logout": 0.05
            },
            # From order details
            "/api/orders/{id}/details": {
                "/api/users/{id}/orders": 0.40,
                "/api/products/browse": 0.35,
                "/api/orders/{id}/track": 0.15,
                "/api/logout": 0.10
            },
            # From order tracking
            "/api/orders/{id}/track": {
                "/api/users/{id}/orders": 0.50,
                "/api/products/browse": 0.30,
                "/api/logout": 0.20
            },
            # From settings
            "/api/users/{id}/settings": {
                "/api/users/{id}/profile": 0.60,
                "/api/products/browse": 0.25,
                "/api/logout": 0.15
            }
        },
        exit_points={
            "/api/logout",
            "/api/confirmation",
            "/api/orders/{id}/track"
        },
        avg_response_times={
            "/api/login": 120,
            "/api/users/{id}/profile": 80,
            "/api/products/browse": 200,
            "/api/products/search": 250,
            "/api/products/{id}/details": 150,
            "/api/products/{id}/reviews": 180,
            "/api/cart": 100,
            "/api/cart/add": 90,
            "/api/cart/remove": 85,
            "/api/checkout": 300,
            "/api/payment": 500,
            "/api/confirmation": 150,
            "/api/users/{id}/orders": 180,
            "/api/orders/{id}/details": 160,
            "/api/orders/{id}/track": 140,
            "/api/users/{id}/settings": 110,
            "/api/logout": 70
        }
    )

    def __init__(self, seed: Optional[int] = None):
        """Initialize the synthetic generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.call_counter = 0

    def generate_session(
        self,
        workflow: WorkflowDefinition,
        user_id: str,
        start_time: datetime,
        user_type: Optional[str] = None,
        max_length: int = 20
    ) -> Session:
        """Generate a single session by walking through the workflow.

        Args:
            workflow: Workflow definition to follow
            user_id: User identifier
            start_time: Session start timestamp
            user_type: User type (premium/free/guest), random if None
            max_length: Maximum number of calls in session

        Returns:
            Generated Session object
        """
        # Determine user type
        if user_type is None:
            user_type = self._sample_user_type(workflow)

        # Start at random entry point
        current_endpoint = self._weighted_choice(workflow.entry_points)

        # Generate session ID
        session_id = f"sess_{user_id}_{int(start_time.timestamp())}"

        # Generate calls
        calls = []
        current_time = start_time

        for step in range(max_length):
            # Generate call
            call = self._generate_call(
                endpoint=current_endpoint,
                user_id=user_id,
                session_id=session_id,
                timestamp=current_time,
                user_type=user_type,
                workflow=workflow
            )
            calls.append(call)

            # Check if we've reached an exit point
            if current_endpoint in workflow.exit_points:
                break

            # Choose next endpoint
            if current_endpoint not in workflow.transitions:
                # Dead end - exit
                break

            next_endpoint = self._weighted_choice(workflow.transitions[current_endpoint])
            current_endpoint = next_endpoint

            # Advance time
            delay = self._generate_delay()
            current_time = current_time + timedelta(seconds=delay)

        # Create session
        session = Session(
            session_id=session_id,
            user_id=user_id,
            user_type=user_type,
            start_timestamp=start_time,
            end_timestamp=current_time,
            calls=calls
        )

        return session

    def generate_dataset(
        self,
        num_users: int,
        sessions_per_user: Tuple[float, float] = (3.0, 2.0),
        workflow: Optional[WorkflowDefinition] = None,
        date_range_days: int = 30,
        start_date: Optional[datetime] = None,
        cascade_failure_rate: float = 0.0,
        show_progress: bool = True
    ) -> Dataset:
        """Generate a complete dataset with many users and sessions.

        Args:
            num_users: Number of unique users to generate
            sessions_per_user: (mean, std) for number of sessions per user
            workflow: Workflow to use (defaults to ECOMMERCE_WORKFLOW)
            date_range_days: Spread sessions over this many days
            start_date: Start date for sessions (defaults to now)
            cascade_failure_rate: Percentage of sessions to inject cascade failures (0.0-1.0)
            show_progress: Show progress bar

        Returns:
            Generated Dataset
        """
        if workflow is None:
            workflow = self.ECOMMERCE_WORKFLOW

        if start_date is None:
            start_date = datetime.now()

        sessions = []

        # Progress tracking
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(range(num_users), desc="Generating users")
            except ImportError:
                iterator = range(num_users)
                print(f"Generating {num_users} users...")
        else:
            iterator = range(num_users)

        for user_idx in iterator:
            user_id = f"user_{user_idx:05d}"

            # Determine number of sessions for this user
            num_sessions = max(1, int(np.random.normal(*sessions_per_user)))

            # Determine user type (consistent across their sessions)
            user_type = self._sample_user_type(workflow)

            # Generate sessions for this user
            for session_idx in range(num_sessions):
                # Random timestamp within date range
                random_day = random.randint(0, date_range_days)
                random_hour = random.randint(0, 23)
                random_minute = random.randint(0, 59)

                session_time = start_date + timedelta(
                    days=random_day,
                    hours=random_hour,
                    minutes=random_minute
                )

                # Generate session
                session = self.generate_session(
                    workflow=workflow,
                    user_id=user_id,
                    start_time=session_time,
                    user_type=user_type
                )

                # Optionally inject cascade failure
                if random.random() < cascade_failure_rate:
                    session = self.inject_cascade(session)

                sessions.append(session)

        # Create dataset
        dataset = Dataset(
            name=f"synthetic_{workflow.name}",
            sessions=sessions,
            metadata={
                'generator': 'SyntheticGenerator',
                'seed': self.seed,
                'workflow': workflow.name,
                'num_users': num_users,
                'date_range_days': date_range_days,
                'cascade_failure_rate': cascade_failure_rate,
                'generation_date': datetime.now().isoformat()
            }
        )

        return dataset

    def inject_cascade(self, session: Session) -> Session:
        """Inject cascade failure symptoms into a session.

        Modifies response times, adds errors, and simulates retries.

        Args:
            session: Original session

        Returns:
            Modified session with cascade failure patterns
        """
        # Create new calls list with failures
        new_calls = []

        # Choose a point where cascade starts (middle of session)
        cascade_start = len(session.calls) // 2

        for idx, call in enumerate(session.calls):
            if idx >= cascade_start:
                # After cascade starts

                # Slow responses
                if random.random() < 0.6:
                    # Multiply response time by 3-10x
                    call.response_time_ms *= random.uniform(3, 10)

                # Add timeouts
                if random.random() < 0.2:
                    call.status_code = 504  # Gateway Timeout
                    call.response_time_ms = 30000  # 30 seconds

                # Add server errors
                if random.random() < 0.15:
                    call.status_code = 503  # Service Unavailable
                    call.response_time_ms *= random.uniform(2, 5)

                # Add retries (duplicate calls)
                new_calls.append(call)
                if random.random() < 0.3:
                    # Add a retry
                    retry_call = APICall(
                        call_id=f"{call.call_id}_retry",
                        endpoint=call.endpoint,
                        method=call.method,
                        params=call.params,
                        user_id=call.user_id,
                        session_id=call.session_id,
                        timestamp=call.timestamp + timedelta(milliseconds=100),
                        response_time_ms=call.response_time_ms * random.uniform(1.5, 3),
                        status_code=call.status_code,
                        response_size_bytes=call.response_size_bytes,
                        user_type=call.user_type
                    )
                    new_calls.append(retry_call)
            else:
                # Before cascade - normal
                new_calls.append(call)

        # Create new session with modified calls
        return Session(
            session_id=session.session_id,
            user_id=session.user_id,
            user_type=session.user_type,
            start_timestamp=session.start_timestamp,
            end_timestamp=session.end_timestamp,
            calls=new_calls
        )

    # Helper methods

    def _sample_user_type(self, workflow: WorkflowDefinition) -> str:
        """Sample a user type based on workflow distribution."""
        return self._weighted_choice(workflow.user_type_distribution)

    def _weighted_choice(self, weights: Dict[str, float]) -> str:
        """Choose an item based on weights."""
        items = list(weights.keys())
        probabilities = [weights[item] for item in items]
        return random.choices(items, weights=probabilities)[0]

    def _generate_call(
        self,
        endpoint: str,
        user_id: str,
        session_id: str,
        timestamp: datetime,
        user_type: str,
        workflow: WorkflowDefinition
    ) -> APICall:
        """Generate a single API call."""
        self.call_counter += 1

        # Determine HTTP method
        if any(keyword in endpoint for keyword in ['/add', '/remove', '/payment', '/checkout']):
            method = "POST"
        elif any(keyword in endpoint for keyword in ['/update', '/settings']):
            method = "PUT"
        elif '/logout' in endpoint:
            method = "POST"
        elif '/login' in endpoint:
            method = "POST"
        else:
            method = "GET"

        # Generate response time
        base_time = workflow.avg_response_times.get(endpoint, 150)
        response_time = max(10, np.random.normal(base_time, base_time * 0.3))

        # Generate response size (bytes)
        if method == "GET":
            response_size = int(np.random.lognormal(8, 1.5))  # 1KB - 50KB
        else:
            response_size = int(np.random.lognormal(6, 1))     # 200B - 5KB

        # Status code (mostly successful)
        if random.random() < 0.95:
            status_code = 200
        elif random.random() < 0.7:
            status_code = 400  # Client error
        else:
            status_code = 500  # Server error

        # Parameters (for search and some endpoints)
        params = {}
        if 'search' in endpoint:
            params = {
                'q': random.choice(['laptop', 'phone', 'tablet', 'camera', 'headphones']),
                'sort': random.choice(['price', 'rating', 'popular'])
            }
        elif 'browse' in endpoint:
            params = {
                'category': random.choice(['electronics', 'clothing', 'books', 'home']),
                'page': str(random.randint(1, 5))
            }

        return APICall(
            call_id=f"call_{self.call_counter:08d}",
            endpoint=endpoint,
            method=method,
            params=params,
            user_id=user_id,
            session_id=session_id,
            timestamp=timestamp,
            response_time_ms=response_time,
            status_code=status_code,
            response_size_bytes=response_size,
            user_type=user_type
        )

    def _generate_delay(self) -> float:
        """Generate delay between calls in seconds."""
        # Most delays are 1-5 seconds with some longer pauses
        if random.random() < 0.9:
            return random.uniform(1, 5)
        else:
            return random.uniform(10, 30)  # Longer pause (reading, thinking)

    @staticmethod
    def validate_workflow(workflow: WorkflowDefinition) -> List[str]:
        """Validate a workflow definition and return any issues.

        Args:
            workflow: Workflow to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Check all referenced endpoints have transitions or are exit points
        all_endpoints = set(workflow.entry_points.keys())
        for trans in workflow.transitions.values():
            all_endpoints.update(trans.keys())

        for endpoint in all_endpoints:
            if endpoint not in workflow.transitions and endpoint not in workflow.exit_points:
                errors.append(f"Endpoint {endpoint} has no transitions and is not an exit point")

        # Check for dead ends
        for endpoint, transitions in workflow.transitions.items():
            reachable = False
            for next_endpoint in transitions.keys():
                if next_endpoint in workflow.transitions or next_endpoint in workflow.exit_points:
                    reachable = True
                    break
            if not reachable:
                errors.append(f"Endpoint {endpoint} leads only to dead ends")

        return errors


# Pre-built workflows

def create_simple_workflow() -> WorkflowDefinition:
    """Create a simple workflow for testing."""
    return WorkflowDefinition(
        name="simple",
        entry_points={
            "/start": 1.0
        },
        transitions={
            "/start": {
                "/middle": 0.7,
                "/end": 0.3
            },
            "/middle": {
                "/end": 1.0
            }
        },
        exit_points={"/end"},
        avg_response_times={
            "/start": 100,
            "/middle": 150,
            "/end": 80
        }
    )


def create_microservices_workflow() -> WorkflowDefinition:
    """Create a workflow simulating microservices architecture."""
    return WorkflowDefinition(
        name="microservices",
        entry_points={
            "/api/gateway": 1.0
        },
        transitions={
            "/api/gateway": {
                "/api/auth/login": 0.6,
                "/api/products/list": 0.3,
                "/api/search": 0.1
            },
            "/api/auth/login": {
                "/api/users/profile": 0.8,
                "/api/products/list": 0.2
            },
            "/api/users/profile": {
                "/api/products/list": 0.5,
                "/api/orders/history": 0.3,
                "/api/users/settings": 0.2
            },
            "/api/products/list": {
                "/api/products/{id}": 0.7,
                "/api/search": 0.2,
                "/api/cart/view": 0.1
            },
            "/api/products/{id}": {
                "/api/cart/add": 0.5,
                "/api/products/list": 0.3,
                "/api/reviews/{id}": 0.2
            },
            "/api/cart/add": {
                "/api/cart/view": 0.8,
                "/api/products/list": 0.2
            },
            "/api/cart/view": {
                "/api/checkout/init": 0.6,
                "/api/products/list": 0.3,
                "/api/cart/remove": 0.1
            },
            "/api/checkout/init": {
                "/api/payment/process": 0.9,
                "/api/cart/view": 0.1
            },
            "/api/payment/process": {
                "/api/orders/confirmation": 1.0
            },
            "/api/orders/history": {
                "/api/orders/{id}": 0.7,
                "/api/products/list": 0.2,
                "/api/auth/logout": 0.1
            },
            "/api/orders/{id}": {
                "/api/orders/history": 0.5,
                "/api/products/list": 0.3,
                "/api/auth/logout": 0.2
            },
            "/api/search": {
                "/api/products/{id}": 0.8,
                "/api/products/list": 0.2
            },
            "/api/users/settings": {
                "/api/users/profile": 0.7,
                "/api/auth/logout": 0.3
            },
            "/api/cart/remove": {
                "/api/cart/view": 0.8,
                "/api/products/list": 0.2
            },
            "/api/reviews/{id}": {
                "/api/products/{id}": 0.5,
                "/api/products/list": 0.5
            }
        },
        exit_points={
            "/api/auth/logout",
            "/api/orders/confirmation"
        },
        avg_response_times={
            "/api/gateway": 50,
            "/api/auth/login": 120,
            "/api/users/profile": 90,
            "/api/products/list": 180,
            "/api/products/{id}": 140,
            "/api/search": 220,
            "/api/cart/view": 100,
            "/api/cart/add": 80,
            "/api/cart/remove": 75,
            "/api/checkout/init": 250,
            "/api/payment/process": 500,
            "/api/orders/confirmation": 150,
            "/api/orders/history": 170,
            "/api/orders/{id}": 140,
            "/api/users/settings": 110,
            "/api/reviews/{id}": 160,
            "/api/auth/logout": 60
        }
    )

