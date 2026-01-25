"""
Payment Service - Payment processing

Handles payment processing, validation, and refunds.
Characteristics: Highest latency (~500ms simulating external gateway), higher failure rate
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


class PaymentService(BaseService):
    """Payment service for transaction processing.

    Simulates external payment gateway with high latency and occasional failures.
    """

    def __init__(self):
        """Initialize the payment service."""
        config = ServiceConfig(
            name="payment-service",
            port=8006,
            base_latency_ms=500,
            latency_std_ms=100,
            failure_rate=0.03,  # Higher failure rate (3%)
            timeout_rate=0.01,  # 1% timeout rate
            endpoints=[
                EndpointConfig(
                    path="/payment/process",
                    method="POST",
                    response_size_bytes=512,
                    latency_multiplier=1.0,
                    description="Process payment"
                ),
                EndpointConfig(
                    path="/payment/validate",
                    method="POST",
                    response_size_bytes=256,
                    latency_multiplier=0.6,
                    description="Validate payment method"
                ),
                EndpointConfig(
                    path="/payment/refund",
                    method="POST",
                    response_size_bytes=384,
                    latency_multiplier=1.2,
                    description="Process refund"
                ),
            ]
        )

        super().__init__(config)

        # In-memory payment records
        self.payments: Dict[str, dict] = {}

        # Register custom endpoints
        self._register_payment_endpoints()

        self.logger.info("PaymentService initialized (simulating external gateway)")

    def _register_payment_endpoints(self):
        """Register payment-specific endpoints."""

        @self.app.post("/payment/process")
        async def process_payment(request: Request):
            """Process a payment transaction.

            Request body:
                {
                    "order_id": "order123",
                    "amount": 599.98,
                    "currency": "USD",
                    "payment_method": {
                        "type": "credit_card",
                        "card_number": "4532********1234",
                        "card_holder": "John Doe",
                        "expiry": "12/26"
                    }
                }

            Response:
                {
                    "status": "success",
                    "transaction_id": "txn_abc123",
                    "amount": 599.98,
                    "currency": "USD",
                    "gateway_response": "approved",
                    "timestamp": "2026-01-25T10:00:00"
                }
            """
            try:
                data = await request.json()
                order_id = data.get("order_id")
                amount = data.get("amount", 0.0)
                currency = data.get("currency", "USD")
                payment_method = data.get("payment_method", {})

                if not order_id or amount <= 0:
                    return {
                        "status": "failed",
                        "error": "Invalid payment request",
                        "error_code": "INVALID_REQUEST"
                    }

                # Simulate random payment failures
                failure_chance = random.random()
                if failure_chance < 0.05:  # 5% chance of payment declined
                    self.logger.warning(f"Payment declined for order {order_id}")
                    return {
                        "status": "failed",
                        "error": "Payment declined by issuer",
                        "error_code": "PAYMENT_DECLINED",
                        "order_id": order_id
                    }

                if failure_chance < 0.08:  # Additional 3% for insufficient funds
                    self.logger.warning(f"Insufficient funds for order {order_id}")
                    return {
                        "status": "failed",
                        "error": "Insufficient funds",
                        "error_code": "INSUFFICIENT_FUNDS",
                        "order_id": order_id
                    }

                # Generate transaction ID
                transaction_id = f"txn_{fake.uuid4()[:8]}"
                timestamp = fake.date_time_this_hour().isoformat()

                # Store payment record
                self.payments[transaction_id] = {
                    "transaction_id": transaction_id,
                    "order_id": order_id,
                    "amount": amount,
                    "currency": currency,
                    "payment_method": payment_method.get("type", "unknown"),
                    "status": "completed",
                    "timestamp": timestamp
                }

                self.logger.info(f"Payment processed: {transaction_id} for order {order_id}, amount: {amount} {currency}")

                return {
                    "status": "success",
                    "transaction_id": transaction_id,
                    "amount": amount,
                    "currency": currency,
                    "gateway_response": "approved",
                    "timestamp": timestamp,
                    "authorization_code": fake.uuid4()[:8].upper()
                }

            except Exception as e:
                self.logger.error(f"Error processing payment: {e}")
                return {
                    "status": "failed",
                    "error": "Payment processing error",
                    "error_code": "PROCESSING_ERROR"
                }

        @self.app.post("/payment/validate")
        async def validate_payment_method(request: Request):
            """Validate a payment method without charging.

            Request body:
                {
                    "payment_method": {
                        "type": "credit_card",
                        "card_number": "4532********1234",
                        "card_holder": "John Doe",
                        "expiry": "12/26",
                        "cvv": "***"
                    }
                }

            Response:
                {
                    "status": "valid",
                    "card_type": "Visa",
                    "can_process": true
                }
            """
            try:
                data = await request.json()
                payment_method = data.get("payment_method", {})

                if not payment_method:
                    return {
                        "status": "invalid",
                        "error": "No payment method provided"
                    }

                method_type = payment_method.get("type", "")

                # Basic validation
                if method_type not in ["credit_card", "debit_card", "paypal", "bank_transfer"]:
                    return {
                        "status": "invalid",
                        "error": "Unsupported payment method"
                    }

                # Simulate card type detection
                card_types = ["Visa", "MasterCard", "American Express", "Discover"]
                card_type = random.choice(card_types)

                # Random validation failure (5%)
                if random.random() < 0.05:
                    return {
                        "status": "invalid",
                        "error": "Card validation failed",
                        "error_code": "INVALID_CARD"
                    }

                self.logger.info(f"Payment method validated: {method_type}")

                return {
                    "status": "valid",
                    "payment_type": method_type,
                    "card_type": card_type if method_type in ["credit_card", "debit_card"] else None,
                    "can_process": True
                }

            except Exception as e:
                self.logger.error(f"Error validating payment method: {e}")
                return {
                    "status": "invalid",
                    "error": "Validation error"
                }

        @self.app.post("/payment/refund")
        async def process_refund(request: Request):
            """Process a refund for a previous transaction.

            Request body:
                {
                    "transaction_id": "txn_abc123",
                    "amount": 599.98,
                    "reason": "Customer request"
                }

            Response:
                {
                    "status": "success",
                    "refund_id": "ref_xyz789",
                    "transaction_id": "txn_abc123",
                    "amount": 599.98,
                    "timestamp": "2026-01-25T10:00:00"
                }
            """
            try:
                data = await request.json()
                transaction_id = data.get("transaction_id")
                amount = data.get("amount", 0.0)
                reason = data.get("reason", "")

                if not transaction_id or amount <= 0:
                    return {
                        "status": "failed",
                        "error": "Invalid refund request"
                    }

                # Check if transaction exists
                transaction = self.payments.get(transaction_id)
                if not transaction:
                    return {
                        "status": "failed",
                        "error": "Transaction not found",
                        "error_code": "TRANSACTION_NOT_FOUND"
                    }

                # Check refund amount
                if amount > transaction["amount"]:
                    return {
                        "status": "failed",
                        "error": "Refund amount exceeds transaction amount",
                        "error_code": "INVALID_AMOUNT"
                    }

                # Generate refund ID
                refund_id = f"ref_{fake.uuid4()[:8]}"
                timestamp = fake.date_time_this_hour().isoformat()

                self.logger.info(f"Refund processed: {refund_id} for transaction {transaction_id}, amount: {amount}")

                return {
                    "status": "success",
                    "refund_id": refund_id,
                    "transaction_id": transaction_id,
                    "amount": amount,
                    "currency": transaction.get("currency", "USD"),
                    "timestamp": timestamp,
                    "reason": reason
                }

            except Exception as e:
                self.logger.error(f"Error processing refund: {e}")
                return {
                    "status": "failed",
                    "error": "Refund processing error",
                    "error_code": "PROCESSING_ERROR"
                }

        @self.app.get("/payment/transaction/{transaction_id}")
        async def get_transaction(transaction_id: str):
            """Get details of a payment transaction.

            Path params:
                transaction_id: Transaction ID

            Response:
                {
                    "transaction_id": "txn_abc123",
                    "order_id": "order123",
                    "amount": 599.98,
                    "currency": "USD",
                    "status": "completed",
                    "timestamp": "2026-01-25T10:00:00"
                }
            """
            try:
                transaction = self.payments.get(transaction_id)

                if not transaction:
                    return {
                        "error": "Transaction not found",
                        "status": "failed"
                    }

                return {**transaction, "status": "success"}

            except Exception as e:
                self.logger.error(f"Error getting transaction: {e}")
                return {
                    "error": "Failed to get transaction",
                    "status": "failed"
                }


def main():
    """Run the payment service."""
    service = PaymentService()
    service.run()


if __name__ == "__main__":
    main()

