"""
Order Service - Order management and processing

Handles order creation, listing, details, and cancellation.
Characteristics: Complex logic, higher latency (~300ms), critical for business
Dependencies: CartService, PaymentService, InventoryService
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from simulator.services.base_service import BaseService, ServiceConfig, EndpointConfig
from faker import Faker
from fastapi import Request
from typing import Dict, List
import random

fake = Faker()


class OrderService(BaseService):
    """Order service for order management.

    Complex service that orchestrates cart, payment, and inventory services.
    """

    def __init__(self):
        """Initialize the order service."""
        config = ServiceConfig(
            name="order-service",
            port=8005,
            base_latency_ms=300,
            latency_std_ms=60,
            failure_rate=0.015,
            timeout_rate=0.005,
            dependencies=["cart-service", "payment-service", "inventory-service"],
            endpoints=[
                EndpointConfig(
                    path="/orders",
                    method="POST",
                    response_size_bytes=1024,
                    latency_multiplier=1.5,
                    dependencies=[
                        "cart-service:/cart",
                        "payment-service:/payment/process",
                        "inventory-service:/inventory/reserve"
                    ],
                    description="Create new order from cart"
                ),
                EndpointConfig(
                    path="/orders",
                    method="GET",
                    response_size_bytes=4096,
                    latency_multiplier=1.0,
                    description="List user's orders"
                ),
                EndpointConfig(
                    path="/orders/{id}",
                    method="GET",
                    response_size_bytes=2048,
                    latency_multiplier=0.8,
                    description="Get order details"
                ),
                EndpointConfig(
                    path="/orders/{id}/cancel",
                    method="POST",
                    response_size_bytes=512,
                    latency_multiplier=1.2,
                    dependencies=["inventory-service:/inventory/release"],
                    description="Cancel order"
                ),
            ]
        )

        super().__init__(config)

        # In-memory order database
        self.orders: Dict[str, dict] = {}
        self.order_statuses = ["pending", "confirmed", "processing", "shipped", "delivered", "cancelled"]

        # Register custom endpoints
        self._register_order_endpoints()

        # Register dependencies
        self.register_service("cart-service", "http://localhost:8004")
        self.register_service("payment-service", "http://localhost:8006")
        self.register_service("inventory-service", "http://localhost:8007")

        self.logger.info("OrderService initialized")

    def _register_order_endpoints(self):
        """Register order-specific endpoints."""

        @self.app.post("/orders")
        async def create_order(request: Request):
            """Create a new order from the user's cart.

            Request body:
                {
                    "user_id": "test_user_001",
                    "payment_method": {
                        "type": "credit_card",
                        "card_number": "4532********1234"
                    },
                    "shipping_address": "123 Main St, City, State 12345"
                }

            Response:
                {
                    "status": "success",
                    "order_id": "order_abc123",
                    "total_amount": 599.98,
                    "payment_status": "completed",
                    "order_status": "confirmed"
                }
            """
            try:
                data = await request.json()
                user_id = data.get("user_id", "test_user_001")
                payment_method = data.get("payment_method", {})
                shipping_address = data.get("shipping_address", "")

                if not payment_method:
                    return {
                        "status": "failed",
                        "error": "Payment method required"
                    }

                # Step 1: Get cart contents
                try:
                    cart = await self.call_service(
                        "cart-service",
                        "/cart",
                        method="GET",
                        params={"user_id": user_id}
                    )

                    if not cart.get("items"):
                        return {
                            "status": "failed",
                            "error": "Cart is empty"
                        }

                    items = cart["items"]
                    total_amount = cart["total_price"]

                except Exception as e:
                    self.logger.error(f"Failed to get cart: {e}")
                    return {
                        "status": "failed",
                        "error": "Failed to retrieve cart"
                    }

                # Step 2: Reserve inventory
                reservation_id = None
                try:
                    inventory_items = [
                        {"product_id": item["product_id"], "quantity": item["quantity"]}
                        for item in items
                    ]

                    order_id_temp = f"order_{fake.uuid4()[:8]}"

                    inventory_result = await self.call_service(
                        "inventory-service",
                        "/inventory/reserve",
                        method="POST",
                        json_data={
                            "order_id": order_id_temp,
                            "items": inventory_items
                        }
                    )

                    if inventory_result.get("status") != "success":
                        return {
                            "status": "failed",
                            "error": "Inventory reservation failed",
                            "details": inventory_result.get("error")
                        }

                    reservation_id = inventory_result.get("reservation_id")

                except Exception as e:
                    self.logger.error(f"Failed to reserve inventory: {e}")
                    return {
                        "status": "failed",
                        "error": "Inventory service unavailable"
                    }

                # Step 3: Process payment
                transaction_id = None
                try:
                    payment_result = await self.call_service(
                        "payment-service",
                        "/payment/process",
                        method="POST",
                        json_data={
                            "order_id": order_id_temp,
                            "amount": total_amount,
                            "currency": "USD",
                            "payment_method": payment_method
                        }
                    )

                    if payment_result.get("status") != "success":
                        # Payment failed - release inventory
                        if reservation_id:
                            try:
                                await self.call_service(
                                    "inventory-service",
                                    "/inventory/release",
                                    method="POST",
                                    json_data={"reservation_id": reservation_id}
                                )
                            except:
                                pass

                        return {
                            "status": "failed",
                            "error": "Payment failed",
                            "details": payment_result.get("error"),
                            "error_code": payment_result.get("error_code")
                        }

                    transaction_id = payment_result.get("transaction_id")

                except Exception as e:
                    self.logger.error(f"Failed to process payment: {e}")
                    # Release inventory
                    if reservation_id:
                        try:
                            await self.call_service(
                                "inventory-service",
                                "/inventory/release",
                                method="POST",
                                json_data={"reservation_id": reservation_id}
                            )
                        except:
                            pass

                    return {
                        "status": "failed",
                        "error": "Payment service unavailable"
                    }

                # Step 4: Create order
                order_id = f"order_{fake.uuid4()[:8]}"
                order = {
                    "order_id": order_id,
                    "user_id": user_id,
                    "items": items,
                    "total_amount": total_amount,
                    "currency": "USD",
                    "shipping_address": shipping_address,
                    "payment_method": payment_method.get("type", "unknown"),
                    "transaction_id": transaction_id,
                    "reservation_id": reservation_id,
                    "order_status": "confirmed",
                    "payment_status": "completed",
                    "created_at": fake.date_time_this_hour().isoformat(),
                    "updated_at": fake.date_time_this_hour().isoformat()
                }

                self.orders[order_id] = order

                # Step 5: Clear cart
                try:
                    await self.call_service(
                        "cart-service",
                        "/cart/clear",
                        method="POST",
                        json_data={"user_id": user_id}
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to clear cart: {e}")

                self.logger.info(f"Order created: {order_id} for user {user_id}, amount: {total_amount}")

                return {
                    "status": "success",
                    "order_id": order_id,
                    "total_amount": total_amount,
                    "currency": "USD",
                    "payment_status": "completed",
                    "order_status": "confirmed",
                    "transaction_id": transaction_id
                }

            except Exception as e:
                self.logger.error(f"Error creating order: {e}")
                return {
                    "status": "failed",
                    "error": "Order creation failed"
                }

        @self.app.get("/orders")
        async def list_orders(user_id: str = "test_user_001", limit: int = 10):
            """List orders for a user.

            Query params:
                user_id: User ID
                limit: Max number of orders to return

            Response:
                {
                    "orders": [...],
                    "count": 5,
                    "user_id": "test_user_001"
                }
            """
            try:
                # Filter orders by user
                user_orders = [
                    order for order in self.orders.values()
                    if order["user_id"] == user_id
                ]

                # Sort by created_at descending
                user_orders.sort(key=lambda x: x["created_at"], reverse=True)

                # Limit results
                user_orders = user_orders[:limit]

                return {
                    "orders": user_orders,
                    "count": len(user_orders),
                    "user_id": user_id,
                    "status": "success"
                }

            except Exception as e:
                self.logger.error(f"Error listing orders: {e}")
                return {
                    "error": "Failed to list orders",
                    "status": "failed"
                }

        @self.app.get("/orders/{order_id}")
        async def get_order(order_id: str):
            """Get detailed information about an order.

            Path params:
                order_id: Order ID

            Response:
                {
                    "order_id": "order_abc123",
                    "user_id": "test_user_001",
                    "items": [...],
                    "total_amount": 599.98,
                    "order_status": "confirmed",
                    "payment_status": "completed",
                    ...
                }
            """
            try:
                order = self.orders.get(order_id)

                if not order:
                    return {
                        "error": "Order not found",
                        "status": "failed"
                    }

                return {**order, "status": "success"}

            except Exception as e:
                self.logger.error(f"Error getting order: {e}")
                return {
                    "error": "Failed to get order",
                    "status": "failed"
                }

        @self.app.post("/orders/{order_id}/cancel")
        async def cancel_order(order_id: str):
            """Cancel an order.

            Path params:
                order_id: Order ID

            Response:
                {
                    "status": "success",
                    "order_id": "order_abc123",
                    "order_status": "cancelled",
                    "message": "Order cancelled successfully"
                }
            """
            try:
                order = self.orders.get(order_id)

                if not order:
                    return {
                        "error": "Order not found",
                        "status": "failed"
                    }

                # Check if order can be cancelled
                if order["order_status"] in ["shipped", "delivered", "cancelled"]:
                    return {
                        "error": f"Cannot cancel order with status '{order['order_status']}'",
                        "status": "failed"
                    }

                # Release inventory reservation
                reservation_id = order.get("reservation_id")
                if reservation_id:
                    try:
                        await self.call_service(
                            "inventory-service",
                            "/inventory/release",
                            method="POST",
                            json_data={"reservation_id": reservation_id}
                        )
                        self.logger.info(f"Released inventory for cancelled order {order_id}")
                    except Exception as e:
                        self.logger.warning(f"Failed to release inventory: {e}")

                # Update order status
                order["order_status"] = "cancelled"
                order["updated_at"] = fake.date_time_this_hour().isoformat()

                self.logger.info(f"Order cancelled: {order_id}")

                return {
                    "status": "success",
                    "order_id": order_id,
                    "order_status": "cancelled",
                    "message": "Order cancelled successfully"
                }

            except Exception as e:
                self.logger.error(f"Error cancelling order: {e}")
                return {
                    "error": "Failed to cancel order",
                    "status": "failed"
                }

        @self.app.get("/orders/stats")
        async def get_order_stats():
            """Get order statistics.

            Response:
                {
                    "total_orders": 50,
                    "total_revenue": 25000.00,
                    "orders_by_status": {...}
                }
            """
            try:
                total_orders = len(self.orders)
                total_revenue = sum(order["total_amount"] for order in self.orders.values())

                # Count by status
                orders_by_status = {}
                for status in self.order_statuses:
                    count = sum(1 for order in self.orders.values()
                               if order["order_status"] == status)
                    orders_by_status[status] = count

                return {
                    "total_orders": total_orders,
                    "total_revenue": round(total_revenue, 2),
                    "orders_by_status": orders_by_status,
                    "status": "success"
                }

            except Exception as e:
                self.logger.error(f"Error getting order stats: {e}")
                return {
                    "error": "Failed to get stats",
                    "status": "failed"
                }


def main():
    """Run the order service."""
    service = OrderService()
    service.run()


if __name__ == "__main__":
    main()

