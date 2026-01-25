"""
Cart Service - Shopping cart management

Handles cart operations: add, update, remove items.
Dependencies: ProductService (validate items), UserService (identify user)
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


class CartService(BaseService):
    """Cart service for shopping cart management.

    Manages user shopping carts with product validation.
    """

    def __init__(self):
        """Initialize the cart service."""
        config = ServiceConfig(
            name="cart-service",
            port=8004,
            base_latency_ms=60,
            latency_std_ms=10,
            failure_rate=0.008,
            timeout_rate=0.001,
            dependencies=["product-service", "user-service"],
            endpoints=[
                EndpointConfig(
                    path="/cart",
                    method="GET",
                    response_size_bytes=2048,
                    latency_multiplier=1.0,
                    description="Get cart contents"
                ),
                EndpointConfig(
                    path="/cart/add",
                    method="POST",
                    response_size_bytes=1024,
                    latency_multiplier=1.2,
                    dependencies=["product-service:/products/{id}"],
                    description="Add item to cart"
                ),
                EndpointConfig(
                    path="/cart/update",
                    method="PUT",
                    response_size_bytes=512,
                    latency_multiplier=1.0,
                    description="Update item quantity"
                ),
                EndpointConfig(
                    path="/cart/remove",
                    method="DELETE",
                    response_size_bytes=256,
                    latency_multiplier=0.8,
                    description="Remove item from cart"
                ),
                EndpointConfig(
                    path="/cart/clear",
                    method="POST",
                    response_size_bytes=128,
                    latency_multiplier=0.7,
                    description="Empty the cart"
                ),
            ]
        )

        super().__init__(config)

        # In-memory cart storage: user_id -> {items: [...]}
        self.carts: Dict[str, dict] = {}

        # Register custom endpoints
        self._register_cart_endpoints()

        # Register dependencies
        self.register_service("product-service", "http://localhost:8003")
        self.register_service("user-service", "http://localhost:8001")

        self.logger.info("CartService initialized")

    def _register_cart_endpoints(self):
        """Register cart-specific endpoints."""

        @self.app.get("/cart")
        async def get_cart(user_id: str = "test_user_001"):
            """Get cart contents for a user.

            Query params:
                user_id: User ID (default: test_user_001)

            Response:
                {
                    "user_id": "test_user_001",
                    "items": [
                        {
                            "product_id": "prod_001",
                            "quantity": 2,
                            "price": 299.99,
                            "subtotal": 599.98
                        }
                    ],
                    "item_count": 1,
                    "total_items": 2,
                    "total_price": 599.98
                }
            """
            try:
                cart = self.carts.get(user_id, {"items": []})
                items = cart.get("items", [])

                # Calculate totals
                total_items = sum(item["quantity"] for item in items)
                total_price = sum(item["subtotal"] for item in items)

                return {
                    "user_id": user_id,
                    "items": items,
                    "item_count": len(items),
                    "total_items": total_items,
                    "total_price": round(total_price, 2),
                    "status": "success"
                }

            except Exception as e:
                self.logger.error(f"Error getting cart for {user_id}: {e}")
                return {
                    "error": "Failed to get cart",
                    "status": "failed"
                }

        @self.app.post("/cart/add")
        async def add_to_cart(request: Request):
            """Add an item to the cart.

            Request body:
                {
                    "user_id": "test_user_001",
                    "product_id": "prod_001",
                    "quantity": 2
                }

            Response:
                {
                    "status": "success",
                    "message": "Item added to cart",
                    "cart_item_count": 3
                }
            """
            try:
                data = await request.json()
                user_id = data.get("user_id", "test_user_001")
                product_id = data.get("product_id")
                quantity = data.get("quantity", 1)

                if not product_id:
                    return {
                        "error": "Missing product_id",
                        "status": "failed"
                    }

                if quantity <= 0:
                    return {
                        "error": "Quantity must be positive",
                        "status": "failed"
                    }

                # Validate product exists
                try:
                    product = await self.call_service(
                        "product-service",
                        f"/products/{product_id}",
                        method="GET"
                    )

                    if product.get("status") == "failed":
                        return {
                            "error": "Product not found",
                            "status": "failed"
                        }

                    # Check if in stock
                    if not product.get("in_stock", True):
                        return {
                            "error": "Product out of stock",
                            "status": "failed"
                        }

                    price = product.get("price", 0.0)

                except Exception as e:
                    self.logger.warning(f"Could not validate product {product_id}: {e}")
                    # Continue with default price
                    price = 99.99

                # Get or create cart
                if user_id not in self.carts:
                    self.carts[user_id] = {"items": []}

                cart = self.carts[user_id]
                items = cart["items"]

                # Check if item already in cart
                existing_item = None
                for item in items:
                    if item["product_id"] == product_id:
                        existing_item = item
                        break

                if existing_item:
                    # Update quantity
                    existing_item["quantity"] += quantity
                    existing_item["subtotal"] = round(
                        existing_item["quantity"] * existing_item["price"], 2
                    )
                else:
                    # Add new item
                    items.append({
                        "product_id": product_id,
                        "quantity": quantity,
                        "price": price,
                        "subtotal": round(quantity * price, 2)
                    })

                self.logger.info(f"Added {quantity}x {product_id} to cart for user {user_id}")

                return {
                    "status": "success",
                    "message": "Item added to cart",
                    "cart_item_count": len(items)
                }

            except Exception as e:
                self.logger.error(f"Error adding to cart: {e}")
                return {
                    "error": "Failed to add item",
                    "status": "failed"
                }

        @self.app.put("/cart/update")
        async def update_cart_item(request: Request):
            """Update quantity of an item in the cart.

            Request body:
                {
                    "user_id": "test_user_001",
                    "product_id": "prod_001",
                    "quantity": 5
                }

            Response:
                {
                    "status": "success",
                    "message": "Cart updated",
                    "new_quantity": 5
                }
            """
            try:
                data = await request.json()
                user_id = data.get("user_id", "test_user_001")
                product_id = data.get("product_id")
                quantity = data.get("quantity", 1)

                if not product_id:
                    return {
                        "error": "Missing product_id",
                        "status": "failed"
                    }

                cart = self.carts.get(user_id)
                if not cart:
                    return {
                        "error": "Cart not found",
                        "status": "failed"
                    }

                # Find item in cart
                item_found = False
                for item in cart["items"]:
                    if item["product_id"] == product_id:
                        if quantity <= 0:
                            # Remove item if quantity is 0 or negative
                            cart["items"].remove(item)
                        else:
                            item["quantity"] = quantity
                            item["subtotal"] = round(quantity * item["price"], 2)
                        item_found = True
                        break

                if not item_found:
                    return {
                        "error": "Item not in cart",
                        "status": "failed"
                    }

                self.logger.info(f"Updated {product_id} to quantity {quantity} for user {user_id}")

                return {
                    "status": "success",
                    "message": "Cart updated",
                    "new_quantity": quantity if quantity > 0 else 0
                }

            except Exception as e:
                self.logger.error(f"Error updating cart: {e}")
                return {
                    "error": "Failed to update cart",
                    "status": "failed"
                }

        @self.app.delete("/cart/remove")
        async def remove_from_cart(user_id: str = "test_user_001", product_id: str = None):
            """Remove an item from the cart.

            Query params:
                user_id: User ID
                product_id: Product ID to remove

            Response:
                {
                    "status": "success",
                    "message": "Item removed from cart"
                }
            """
            try:
                if not product_id:
                    return {
                        "error": "Missing product_id",
                        "status": "failed"
                    }

                cart = self.carts.get(user_id)
                if not cart:
                    return {
                        "error": "Cart not found",
                        "status": "failed"
                    }

                # Remove item
                original_count = len(cart["items"])
                cart["items"] = [item for item in cart["items"]
                                if item["product_id"] != product_id]

                if len(cart["items"]) == original_count:
                    return {
                        "error": "Item not in cart",
                        "status": "failed"
                    }

                self.logger.info(f"Removed {product_id} from cart for user {user_id}")

                return {
                    "status": "success",
                    "message": "Item removed from cart"
                }

            except Exception as e:
                self.logger.error(f"Error removing from cart: {e}")
                return {
                    "error": "Failed to remove item",
                    "status": "failed"
                }

        @self.app.post("/cart/clear")
        async def clear_cart(request: Request):
            """Clear all items from the cart.

            Request body:
                {
                    "user_id": "test_user_001"
                }

            Response:
                {
                    "status": "success",
                    "message": "Cart cleared"
                }
            """
            try:
                data = await request.json()
                user_id = data.get("user_id", "test_user_001")

                if user_id in self.carts:
                    self.carts[user_id] = {"items": []}
                    self.logger.info(f"Cleared cart for user {user_id}")

                return {
                    "status": "success",
                    "message": "Cart cleared"
                }

            except Exception as e:
                self.logger.error(f"Error clearing cart: {e}")
                return {
                    "error": "Failed to clear cart",
                    "status": "failed"
                }


def main():
    """Run the cart service."""
    service = CartService()
    service.run()


if __name__ == "__main__":
    main()

