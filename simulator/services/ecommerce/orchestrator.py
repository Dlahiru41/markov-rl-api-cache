"""
E-commerce Service Orchestrator

Manages multiple microservices - start, stop, and monitor them all together.
"""

import subprocess
import time
import signal
import sys
import httpx
from typing import List, Dict
import threading


class ServiceOrchestrator:
    """Orchestrates multiple microservices."""

    def __init__(self):
        """Initialize the orchestrator."""
        self.services: Dict[str, dict] = {}
        self.processes: Dict[str, subprocess.Popen] = {}

        # Define all services
        self.service_definitions = [
            {
                "name": "auth-service",
                "module": "simulator.services.ecommerce.auth_service",
                "port": 8002,
                "startup_time": 2
            },
            {
                "name": "user-service",
                "module": "simulator.services.ecommerce.user_service",
                "port": 8001,
                "startup_time": 3,
                "depends_on": ["auth-service"]
            },
            {
                "name": "inventory-service",
                "module": "simulator.services.ecommerce.inventory_service",
                "port": 8007,
                "startup_time": 2
            },
            {
                "name": "product-service",
                "module": "simulator.services.ecommerce.product_service",
                "port": 8003,
                "startup_time": 3,
                "depends_on": ["inventory-service"]
            },
            {
                "name": "cart-service",
                "module": "simulator.services.ecommerce.cart_service",
                "port": 8004,
                "startup_time": 2,
                "depends_on": ["product-service"]
            },
            {
                "name": "payment-service",
                "module": "simulator.services.ecommerce.payment_service",
                "port": 8006,
                "startup_time": 2
            },
            {
                "name": "order-service",
                "module": "simulator.services.ecommerce.order_service",
                "port": 8005,
                "startup_time": 3,
                "depends_on": ["cart-service", "payment-service", "inventory-service"]
            },
        ]

    def start_service(self, service_def: dict):
        """Start a single service.

        Args:
            service_def: Service definition dictionary
        """
        name = service_def["name"]
        module = service_def["module"]
        port = service_def["port"]

        print(f"Starting {name} on port {port}...")

        if sys.platform == "win32":
            process = subprocess.Popen(
                ["python", "-m", module],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )
        else:
            process = subprocess.Popen(
                ["python", "-m", module],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

        self.processes[name] = process
        self.services[name] = service_def

        print(f"  ✓ {name} started (PID: {process.pid})")

    def wait_for_service(self, name: str, port: int, timeout: int = 10):
        """Wait for a service to be ready.

        Args:
            name: Service name
            port: Service port
            timeout: Max time to wait in seconds
        """
        url = f"http://localhost:{port}/health"
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = httpx.get(url, timeout=2.0)
                if response.status_code == 200:
                    print(f"  ✓ {name} is ready")
                    return True
            except:
                time.sleep(0.5)

        print(f"  ⚠ {name} did not respond within {timeout}s")
        return False

    def start_all(self):
        """Start all services in dependency order."""
        print("\n" + "=" * 70)
        print("STARTING E-COMMERCE MICROSERVICES")
        print("=" * 70 + "\n")

        # Start services in order
        for service_def in self.service_definitions:
            self.start_service(service_def)
            time.sleep(service_def.get("startup_time", 2))
            self.wait_for_service(service_def["name"], service_def["port"])

        print("\n" + "=" * 70)
        print("✓ ALL SERVICES STARTED")
        print("=" * 70)
        self.print_status()

    def stop_all(self):
        """Stop all services."""
        print("\n" + "=" * 70)
        print("STOPPING ALL SERVICES")
        print("=" * 70 + "\n")

        for name, process in self.processes.items():
            print(f"Stopping {name}...")
            try:
                if sys.platform == "win32":
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                else:
                    process.terminate()
                    process.wait(timeout=5)
                print(f"  ✓ {name} stopped")
            except Exception as e:
                print(f"  ⚠ Error stopping {name}: {e}")

        self.processes.clear()
        print("\n✓ All services stopped")

    def print_status(self):
        """Print status of all services."""
        print("\n" + "-" * 70)
        print("SERVICE STATUS")
        print("-" * 70)
        print(f"{'Service':<25} {'Port':<10} {'Status':<15} {'PID':<10}")
        print("-" * 70)

        for name, service_def in self.services.items():
            port = service_def["port"]
            process = self.processes.get(name)

            if process and process.poll() is None:
                status = "Running"
                pid = process.pid
            else:
                status = "Stopped"
                pid = "-"

            print(f"{name:<25} {port:<10} {status:<15} {pid:<10}")

        print("-" * 70)
        print()

    def health_check_all(self):
        """Check health of all services."""
        print("\n" + "-" * 70)
        print("HEALTH CHECK")
        print("-" * 70)

        for name, service_def in self.services.items():
            port = service_def["port"]
            url = f"http://localhost:{port}/health"

            try:
                response = httpx.get(url, timeout=2.0)
                if response.status_code == 200:
                    data = response.json()
                    status = data.get("status", "unknown")
                    print(f"{name:<25} ✓ {status}")
                else:
                    print(f"{name:<25} ✗ HTTP {response.status_code}")
            except Exception as e:
                print(f"{name:<25} ✗ Not responding")

        print("-" * 70)
        print()

    def run_interactive(self):
        """Run in interactive mode with command prompt."""
        try:
            self.start_all()

            print("\nInteractive Mode - Commands:")
            print("  status  - Show service status")
            print("  health  - Check service health")
            print("  stop    - Stop all services and exit")
            print()

            while True:
                try:
                    cmd = input("orchestrator> ").strip().lower()

                    if cmd == "status":
                        self.print_status()
                    elif cmd == "health":
                        self.health_check_all()
                    elif cmd == "stop":
                        break
                    elif cmd == "help":
                        print("Commands: status, health, stop, help")
                    elif cmd:
                        print(f"Unknown command: {cmd}")
                        print("Type 'help' for available commands")
                except EOFError:
                    break

        finally:
            self.stop_all()


def main():
    """Main entry point."""
    orchestrator = ServiceOrchestrator()

    # Set up signal handlers
    def signal_handler(sig, frame):
        print("\n\nReceived interrupt signal...")
        orchestrator.stop_all()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)

    # Run
    try:
        orchestrator.run_interactive()
    except Exception as e:
        print(f"\nError: {e}")
        orchestrator.stop_all()
        sys.exit(1)


if __name__ == "__main__":
    main()

