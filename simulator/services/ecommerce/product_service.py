"""
Product Service - Product catalog and search

Handles product listings, details, search, and recommendations.
Characteristics: Medium latency, search is slower (~200ms)
Dependencies: InventoryService for stock checks
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


class ProductService(BaseService):
    """Product service for catalog management.

    Provides product listings, search, and recommendations with inventory integration.
    """

    def __init__(self):
        """Initialize the product service."""
        config = ServiceConfig(
            name="product-service",
            port=8003,
            base_latency_ms=80,
            latency_std_ms=15,
            failure_rate=0.01,
            timeout_rate=0.002,
            dependencies=["inventory-service"],
            endpoints=[
                EndpointConfig(
                    path="/products",
                    method="GET",
                    response_size_bytes=4096,
                    latency_multiplier=1.0,
                    description="List products with pagination"
                ),
                EndpointConfig(
                    path="/products/{id}",
                    method="GET",
                    response_size_bytes=2048,
                    latency_multiplier=0.8,
                    dependencies=["inventory-service:/inventory/{product_id}"],
                    description="Get single product details"
                ),
                EndpointConfig(
                    path="/search",
                    method="GET",
                    response_size_bytes=8192,
                    latency_multiplier=2.5,
                    description="Search products (slow due to search complexity)"
                ),
                EndpointConfig(
                    path="/recommendations",
                    method="GET",
                    response_size_bytes=4096,
                    latency_multiplier=2.0,
                    description="Get personalized recommendations"
                ),
            ]
        )

        super().__init__(config)

        # In-memory product database
        self.products: Dict[str, dict] = {}
        self.categories = ["Electronics", "Clothing", "Books", "Home & Garden", "Sports", "Toys"]

        # Create fake products
        self._create_fake_products()

        # Register custom endpoints
        self._register_product_endpoints()

        # Register inventory service
        self.register_service("inventory-service", "http://localhost:8007")

        self.logger.info(f"ProductService initialized with {len(self.products)} products")

    def _create_fake_products(self):
        """Create fake product data."""
        for i in range(100):
            product_id = f"prod_{i:03d}"
            category = random.choice(self.categories)
            self.products[product_id] = {
                "product_id": product_id,
                "name": fake.catch_phrase(),
                "description": fake.text(max_nb_chars=200),
                "category": category,
                "price": round(random.uniform(9.99, 999.99), 2),
                "brand": fake.company(),
                "rating": round(random.uniform(3.0, 5.0), 1),
                "reviews_count": random.randint(0, 500),
                "image_url": f"https://example.com/images/{product_id}.jpg",
                "created_at": fake.date_time_this_year().isoformat()
            }

    def _register_product_endpoints(self):
        """Register product-specific endpoints."""

        @self.app.get("/products")
        async def list_products(
            page: int = 1,
            page_size: int = 20,
            category: str = None
        ):
            """List products with pagination.

            Query params:
                page: Page number (default 1)
                page_size: Items per page (default 20)
                category: Filter by category (optional)

            Response:
                {
                    "products": [...],
                    "page": 1,
                    "page_size": 20,
                    "total": 100,
                    "total_pages": 5
                }
            """
            try:
                # Filter by category if specified
                if category:
                    products = [p for p in self.products.values() if p["category"] == category]
                else:
                    products = list(self.products.values())

                # Sort by product_id for consistency
                products.sort(key=lambda x: x["product_id"])

                # Paginate
                total = len(products)
                start = (page - 1) * page_size
                end = start + page_size
                page_products = products[start:end]

                return {
                    "products": page_products,
                    "page": page,
                    "page_size": page_size,
                    "total": total,
                    "total_pages": (total + page_size - 1) // page_size,
                    "status": "success"
                }

            except Exception as e:
                self.logger.error(f"Error listing products: {e}")
                return {
                    "error": "Failed to list products",
                    "status": "failed"
                }

        @self.app.get("/products/{product_id}")
        async def get_product(product_id: str):
            """Get detailed information about a single product.

            Path params:
                product_id: Product ID

            Response:
                {
                    "product_id": "prod_001",
                    "name": "Amazing Product",
                    "description": "...",
                    "category": "Electronics",
                    "price": 299.99,
                    "brand": "BrandName",
                    "rating": 4.5,
                    "reviews_count": 123,
                    "in_stock": true,
                    "available_quantity": 50
                }
            """
            try:
                product = self.products.get(product_id)

                if not product:
                    return {
                        "error": "Product not found",
                        "status": "failed"
                    }

                # Get inventory information
                try:
                    inventory = await self.call_service(
                        "inventory-service",
                        f"/inventory/{product_id}",
                        method="GET"
                    )

                    in_stock = inventory.get("in_stock", False)
                    available_quantity = inventory.get("available", 0)
                except Exception as e:
                    self.logger.warning(f"Could not get inventory for {product_id}: {e}")
                    in_stock = True
                    available_quantity = 100  # Default fallback

                # Combine product and inventory data
                result = {**product}
                result["in_stock"] = in_stock
                result["available_quantity"] = available_quantity
                result["status"] = "success"

                return result

            except Exception as e:
                self.logger.error(f"Error getting product {product_id}: {e}")
                return {
                    "error": "Failed to get product",
                    "status": "failed"
                }

        @self.app.get("/search")
        async def search_products(
            q: str = "",
            category: str = None,
            min_price: float = None,
            max_price: float = None,
            min_rating: float = None
        ):
            """Search products (slower operation due to search complexity).

            Query params:
                q: Search query
                category: Filter by category
                min_price: Minimum price
                max_price: Maximum price
                min_rating: Minimum rating

            Response:
                {
                    "results": [...],
                    "count": 15,
                    "query": "laptop"
                }
            """
            try:
                results = []

                for product in self.products.values():
                    # Text search
                    if q and q.lower() not in product["name"].lower() and \
                       q.lower() not in product["description"].lower():
                        continue

                    # Category filter
                    if category and product["category"] != category:
                        continue

                    # Price filters
                    if min_price is not None and product["price"] < min_price:
                        continue
                    if max_price is not None and product["price"] > max_price:
                        continue

                    # Rating filter
                    if min_rating is not None and product["rating"] < min_rating:
                        continue

                    results.append(product)

                # Sort by relevance (rating for simplicity)
                results.sort(key=lambda x: x["rating"], reverse=True)

                self.logger.info(f"Search query '{q}' returned {len(results)} results")

                return {
                    "results": results,
                    "count": len(results),
                    "query": q,
                    "status": "success"
                }

            except Exception as e:
                self.logger.error(f"Error searching products: {e}")
                return {
                    "error": "Search failed",
                    "status": "failed"
                }

        @self.app.get("/recommendations")
        async def get_recommendations(
            user_id: str = None,
            product_id: str = None,
            limit: int = 10
        ):
            """Get personalized product recommendations.

            Query params:
                user_id: User ID for personalization
                product_id: Product ID for similar products
                limit: Number of recommendations (default 10)

            Response:
                {
                    "recommendations": [...],
                    "count": 10,
                    "algorithm": "collaborative_filtering"
                }
            """
            try:
                # Simple recommendation: random products with high ratings
                candidates = [p for p in self.products.values() if p["rating"] >= 4.0]

                # If product_id provided, recommend from same category
                if product_id and product_id in self.products:
                    category = self.products[product_id]["category"]
                    candidates = [p for p in candidates if p["category"] == category]

                # Randomly select recommendations
                recommendations = random.sample(
                    candidates,
                    min(limit, len(candidates))
                )

                return {
                    "recommendations": recommendations,
                    "count": len(recommendations),
                    "algorithm": "collaborative_filtering",
                    "status": "success"
                }

            except Exception as e:
                self.logger.error(f"Error getting recommendations: {e}")
                return {
                    "error": "Failed to get recommendations",
                    "status": "failed"
                }

        @self.app.get("/categories")
        async def get_categories():
            """Get list of all product categories.

            Response:
                {
                    "categories": ["Electronics", "Clothing", ...],
                    "count": 6
                }
            """
            return {
                "categories": self.categories,
                "count": len(self.categories),
                "status": "success"
            }


def main():
    """Run the product service."""
    service = ProductService()
    service.run()


if __name__ == "__main__":
    main()

