"""
Inventory Service - Fast stock management

Handles inventory checks, reservations, and releases.
Characteristics: Fast, simple operations (~30ms)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from simulator.services.base_service import BaseService, ServiceConfig, EndpointConfig
from faker import Faker
from fastapi import Request
from typing import Dict
import random

fake = Faker()


class InventoryService(BaseService):
    """Inventory service for stock management.

    Fast service with simple operations for checking and managing stock levels.
    """

    def __init__(self):
        """Initialize the inventory service."""
        config = ServiceConfig(
            name="inventory-service",
            port=8007,
            base_latency_ms=30,
            latency_std_ms=5,
            failure_rate=0.005,
            timeout_rate=0.001,
            endpoints=[
                EndpointConfig(
                    path="/inventory/{product_id}",
                    method="GET",
                    response_size_bytes=256,
                    latency_multiplier=1.0,
                    description="Check stock level for a product"
                ),
                EndpointConfig(
                    path="/inventory/reserve",
                    method="POST",
                    response_size_bytes=384,
                    latency_multiplier=1.2,
                    description="Reserve items for an order"
                ),
                EndpointConfig(
                    path="/inventory/release",
                    method="POST",
                    response_size_bytes=256,
                    latency_multiplier=1.0,
                    description="Release reserved items"
                ),
            ]
        )

        super().__init__(config)

        # In-memory inventory database
        self.inventory: Dict[str, dict] = {}
        self.reservations: Dict[str, dict] = {}  # reservation_id -> {product_id, quantity}

        # Create fake inventory
        self._create_fake_inventory()

        # Register custom endpoints
        self._register_inventory_endpoints()

        self.logger.info(f"InventoryService initialized with {len(self.inventory)} products")

    def _create_fake_inventory(self):
        """Create fake inventory data."""
        for i in range(100):
            product_id = f"prod_{i:03d}"
            self.inventory[product_id] = {
                "product_id": product_id,
                "quantity": random.randint(0, 500),
                "reserved": 0,
                "warehouse": fake.city(),
                "last_updated": fake.date_time_this_month().isoformat()
            }

    def _register_inventory_endpoints(self):
        """Register inventory-specific endpoints."""

        @self.app.get("/inventory/{product_id}")
        async def get_inventory(product_id: str):
            """Get stock level for a product.

            Path params:
                product_id: Product ID

            Response:
                {
                    "product_id": "prod_001",
                    "quantity": 150,
                    "available": 145,
                    "reserved": 5,
                    "in_stock": true,
                    "warehouse": "New York"
                }
            """
            try:
                inventory = self.inventory.get(product_id)

                if not inventory:
                    # Create on-demand if doesn't exist
                    inventory = {
                        "product_id": product_id,
                        "quantity": random.randint(0, 200),
                        "reserved": 0,
                        "warehouse": fake.city(),
                        "last_updated": fake.date_time_this_month().isoformat()
                    }
                    self.inventory[product_id] = inventory

                available = inventory["quantity"] - inventory["reserved"]

                return {
                    "product_id": product_id,
                    "quantity": inventory["quantity"],
                    "available": available,
                    "reserved": inventory["reserved"],
                    "in_stock": available > 0,
                    "warehouse": inventory["warehouse"],
                    "status": "success"
                }

            except Exception as e:
                self.logger.error(f"Error getting inventory for {product_id}: {e}")
                return {
                    "error": "Failed to get inventory",
                    "status": "failed"
                }

        @self.app.post("/inventory/reserve")
        async def reserve_inventory(request: Request):
            """Reserve items for an order.

            Request body:
                {
                    "order_id": "order123",
                    "items": [
                        {"product_id": "prod_001", "quantity": 2},
                        {"product_id": "prod_002", "quantity": 1}
                    ]
                }

            Response:
                {
                    "status": "success",
                    "reservation_id": "res_abc123",
                    "reserved_items": [
                        {"product_id": "prod_001", "quantity": 2, "status": "reserved"},
                        {"product_id": "prod_002", "quantity": 1, "status": "reserved"}
                    ]
                }
            """
            try:
                data = await request.json()
                order_id = data.get("order_id")
                items = data.get("items", [])

                if not order_id or not items:
                    return {
                        "error": "Missing order_id or items",
                        "status": "failed"
                    }

                # Check if all items can be reserved
                reserved_items = []
                insufficient_stock = []

                for item in items:
                    product_id = item["product_id"]
                    quantity = item["quantity"]

                    inventory = self.inventory.get(product_id)
                    if not inventory:
                        insufficient_stock.append({
                            "product_id": product_id,
                            "requested": quantity,
                            "available": 0
                        })
                        continue

                    available = inventory["quantity"] - inventory["reserved"]
                    if available < quantity:
                        insufficient_stock.append({
                            "product_id": product_id,
                            "requested": quantity,
                            "available": available
                        })
                    else:
                        reserved_items.append({
                            "product_id": product_id,
                            "quantity": quantity,
                            "status": "reserved"
                        })

                # If any item has insufficient stock, fail the entire reservation
                if insufficient_stock:
                    return {
                        "status": "failed",
                        "error": "Insufficient stock",
                        "insufficient_items": insufficient_stock
                    }

                # Reserve all items
                reservation_id = f"res_{fake.uuid4()[:8]}"
                for item in reserved_items:
                    product_id = item["product_id"]
                    quantity = item["quantity"]
                    self.inventory[product_id]["reserved"] += quantity

                # Store reservation
                self.reservations[reservation_id] = {
                    "order_id": order_id,
                    "items": reserved_items,
                    "created_at": fake.date_time_this_hour().isoformat()
                }

                self.logger.info(f"Reserved {len(reserved_items)} items for order {order_id}")

                return {
                    "status": "success",
                    "reservation_id": reservation_id,
                    "reserved_items": reserved_items
                }

            except Exception as e:
                self.logger.error(f"Error reserving inventory: {e}")
                return {
                    "error": "Reservation failed",
                    "status": "failed"
                }

        @self.app.post("/inventory/release")
        async def release_inventory(request: Request):
            """Release reserved items (e.g., when order is cancelled).

            Request body:
                {
                    "reservation_id": "res_abc123"
                }

            Response:
                {
                    "status": "success",
                    "released_items": [
                        {"product_id": "prod_001", "quantity": 2},
                        {"product_id": "prod_002", "quantity": 1}
                    ]
                }
            """
            try:
                data = await request.json()
                reservation_id = data.get("reservation_id")

                if not reservation_id:
                    return {
                        "error": "Missing reservation_id",
                        "status": "failed"
                    }

                reservation = self.reservations.get(reservation_id)
                if not reservation:
                    return {
                        "error": "Reservation not found",
                        "status": "failed"
                    }

                # Release items
                released_items = []
                for item in reservation["items"]:
                    product_id = item["product_id"]
                    quantity = item["quantity"]

                    if product_id in self.inventory:
                        self.inventory[product_id]["reserved"] -= quantity
                        released_items.append({
                            "product_id": product_id,
                            "quantity": quantity
                        })

                # Remove reservation
                del self.reservations[reservation_id]

                self.logger.info(f"Released reservation {reservation_id}")

                return {
                    "status": "success",
                    "released_items": released_items
                }

            except Exception as e:
                self.logger.error(f"Error releasing inventory: {e}")
                return {
                    "error": "Release failed",
                    "status": "failed"
                }

        @self.app.get("/inventory/stats")
        async def get_inventory_stats():
            """Get overall inventory statistics.

            Response:
                {
                    "total_products": 100,
                    "total_quantity": 25000,
                    "total_reserved": 150,
                    "out_of_stock_count": 5
                }
            """
            try:
                total_products = len(self.inventory)
                total_quantity = sum(inv["quantity"] for inv in self.inventory.values())
                total_reserved = sum(inv["reserved"] for inv in self.inventory.values())
                out_of_stock = sum(1 for inv in self.inventory.values()
                                  if inv["quantity"] - inv["reserved"] <= 0)

                return {
                    "total_products": total_products,
                    "total_quantity": total_quantity,
                    "total_reserved": total_reserved,
                    "out_of_stock_count": out_of_stock,
                    "status": "success"
                }

            except Exception as e:
                self.logger.error(f"Error getting inventory stats: {e}")
                return {
                    "error": "Failed to get stats",
                    "status": "failed"
                }


def main():
    """Run the inventory service."""
    service = InventoryService()
    service.run()


if __name__ == "__main__":
    main()

