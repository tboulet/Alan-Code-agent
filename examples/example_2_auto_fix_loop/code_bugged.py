"""Inventory management module — contains several bugs.

This module provides a Product class and an Inventory manager for tracking
products, computing values, applying discounts, and generating reports.
The code has 5 deliberate bugs that the tests will expose.
"""

from datetime import datetime
from typing import Optional


class Product:
    """A product with name, price, quantity, and optional category."""

    def __init__(self, name: str, price: float, quantity: int, category: str = "general"):
        self.name = name
        self.price = price
        self.quantity = quantity
        self.category = category
        self.created_at = datetime.now()
        self.discount_history: list[float] = []

    def total_value(self) -> float:
        """Return the total value of this product in stock."""
        return self.price + self.quantity  # BUG 1: should be price * quantity

    def apply_discount(self, percent: float) -> float:
        """Apply a percentage discount. Returns the new price.

        Args:
            percent: Discount percentage (e.g., 20 for 20% off).
        """
        discount_amount = self.price * percent  # BUG 2: percent not divided by 100
        self.price -= discount_amount
        self.discount_history.append(percent)
        return self.price

    def __repr__(self) -> str:
        return f"Product({self.name!r}, ${self.price:.2f}, qty={self.quantity})"


class Inventory:
    """Manages a collection of products."""

    def __init__(self):
        self.products: dict[str, Product] = {}
        self._transaction_log: list[str] = []

    def add_product(self, name: str, price: float, quantity: int, category: str = "general") -> Product:
        """Add a product or increase quantity if it already exists."""
        if name in self.products:
            self.products[name].quantity += quantity
            self._log(f"Restocked {name}: +{quantity}")
        else:
            self.products[name] = Product(name, price, quantity, category)
            self._log(f"Added {name}: ${price} x{quantity}")
        return self.products[name]

    def remove_product(self, name: str) -> Product:
        """Remove a product entirely. Raises KeyError if not found."""
        product = self.products.pop(name)
        self._log(f"Removed {name}")
        return product

    def restock(self, name: str, quantity: int) -> int:
        """Add quantity to an existing product. Returns new quantity."""
        product = self.products[name]
        product.quantity = quantity  # BUG 3: should be += not =
        self._log(f"Restocked {name}: +{quantity}")
        return product.quantity

    def total_inventory_value(self) -> float:
        """Calculate total value across all products."""
        total = 0.0
        for product in self.products:  # BUG 4: iterates over keys (str), not values
            total += product.total_value()
        return total

    def find_products_by_category(self, category: str) -> list[Product]:
        """Find all products in a given category."""
        return [p for p in self.products.values() if p.category == category]

    def find_low_stock(self, threshold: int) -> list[Product]:
        """Find products with quantity at or below the threshold."""
        result = []
        for product in self.products.values():
            if product.quantity < threshold:  # BUG 5: should be <= for "at or below"
                result.append(product)
        return result

    def generate_report(self) -> str:
        """Generate a text summary of the inventory."""
        lines = [
            "=== Inventory Report ===",
            f"Total products: {len(self.products)}",
        ]

        by_category: dict[str, list[Product]] = {}
        for p in self.products.values():
            by_category.setdefault(p.category, []).append(p)

        for cat, products in sorted(by_category.items()):
            cat_value = sum(p.total_value() for p in products)
            lines.append(f"\n[{cat}] ({len(products)} items, value: ${cat_value:.2f})")
            for p in sorted(products, key=lambda x: x.name):
                lines.append(f"  - {p.name}: ${p.price:.2f} x{p.quantity} = ${p.total_value():.2f}")

        lines.append(f"\nTotal inventory value: ${self.total_inventory_value():.2f}")
        lines.append(f"Transaction log: {len(self._transaction_log)} entries")
        return "\n".join(lines)

    def _log(self, message: str) -> None:
        self._transaction_log.append(f"[{datetime.now().isoformat()}] {message}")
