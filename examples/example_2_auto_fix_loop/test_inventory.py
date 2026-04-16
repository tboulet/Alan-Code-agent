"""Tests for the inventory module.

These tests are correct — the bugs are in code_bugged.py / code_fixed.py.
DO NOT MODIFY THIS FILE.
"""

import sys
from pathlib import Path

# Import from code_fixed.py (the file Alan is supposed to create)
# This ensures Alan must create a fixed copy, not modify the original.
sys.path.insert(0, str(Path(__file__).parent))
from code_fixed import Product, Inventory


def test_product_total_value():
    """Product value = price * quantity."""
    p = Product("Widget", price=10.0, quantity=5)
    assert p.total_value() == 50.0, f"Expected 50.0, got {p.total_value()}"


def test_product_discount():
    """20% discount on $100 product → $80."""
    p = Product("Gadget", price=100.0, quantity=1)
    new_price = p.apply_discount(20)
    assert new_price == 80.0, f"Expected 80.0 after 20% discount, got {new_price}"
    assert len(p.discount_history) == 1


def test_add_product_new():
    """Adding a new product stores it correctly."""
    inv = Inventory()
    p = inv.add_product("Widget", 10.0, 5)
    assert p.name == "Widget"
    assert p.quantity == 5
    assert p.price == 10.0


def test_add_product_existing():
    """Adding an existing product increases quantity."""
    inv = Inventory()
    inv.add_product("Widget", 10.0, 5)
    p = inv.add_product("Widget", 10.0, 3)
    assert p.quantity == 8, f"Expected 8 after adding 3 to 5, got {p.quantity}"


def test_restock():
    """Restocking adds to existing quantity, not replaces."""
    inv = Inventory()
    inv.add_product("Widget", 10.0, 5)
    new_qty = inv.restock("Widget", 3)
    assert new_qty == 8, f"Expected 8 (5+3), got {new_qty}"


def test_total_inventory_value():
    """Total value = sum of (price * quantity) for each product."""
    inv = Inventory()
    inv.add_product("Widget", 10.0, 5)   # value = 50
    inv.add_product("Gadget", 20.0, 3)   # value = 60
    total = inv.total_inventory_value()
    assert total == 110.0, f"Expected 110.0, got {total}"


def test_find_low_stock():
    """Find products with quantity AT OR BELOW threshold."""
    inv = Inventory()
    inv.add_product("Widget", 10.0, 5)
    inv.add_product("Gadget", 20.0, 2)
    inv.add_product("Doohickey", 5.0, 10)
    low = inv.find_low_stock(5)
    names = {p.name for p in low}
    assert names == {"Widget", "Gadget"}, f"Expected Widget and Gadget (qty <= 5), got {names}"


def test_find_by_category():
    """Category filtering works correctly."""
    inv = Inventory()
    inv.add_product("Wrench", 15.0, 10, category="tools")
    inv.add_product("Hammer", 25.0, 5, category="tools")
    inv.add_product("Apple", 1.5, 100, category="food")
    tools = inv.find_products_by_category("tools")
    assert len(tools) == 2
    assert {p.name for p in tools} == {"Wrench", "Hammer"}


def test_generate_report():
    """Report includes all products and total value."""
    inv = Inventory()
    inv.add_product("Widget", 10.0, 5, category="parts")
    inv.add_product("Gadget", 20.0, 3, category="electronics")
    report = inv.generate_report()
    assert "Widget" in report
    assert "Gadget" in report
    assert "Total inventory value" in report


# ── Runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_product_total_value,
        test_product_discount,
        test_add_product_new,
        test_add_product_existing,
        test_restock,
        test_total_inventory_value,
        test_find_low_stock,
        test_find_by_category,
        test_generate_report,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
            print(f"  PASS: {test.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL: {test.__name__}: {e}")
    print(f"\nResults: {passed} passed, {failed} failed out of {len(tests)}")
    sys.exit(0 if failed == 0 else 1)
