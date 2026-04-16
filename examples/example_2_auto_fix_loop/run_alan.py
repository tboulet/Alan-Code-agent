"""Auto-fix loop: Alan fixes bugs iteratively until tests pass.

Demonstrates using AlanCodeAgent as a library — no async, no CLI, just a
simple Python script that calls agent.query() in a loop.

The script:
1. Asks Alan to read code_bugged.py and create a fixed code_fixed.py
2. Runs the tests
3. If tests fail, sends the output back to Alan
4. Repeats until all tests pass or max attempts reached

Usage:
    # With OpenRouter (needs OPENROUTER_API_KEY env var):
    python examples/example_2_auto_fix_loop/run_alan.py

    # With a specific model:
    python examples/example_2_auto_fix_loop/run_alan.py --model openrouter/google/gemini-2.5-flash

    # With scripted provider (no API, for demonstration):
    python examples/example_2_auto_fix_loop/run_alan.py --scripted
"""

import os
import subprocess
import sys
from pathlib import Path

# Ensure alancode is importable from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from alancode import AlanCodeAgent

EXAMPLE_DIR = Path(__file__).resolve().parent
MAX_ATTEMPTS = 5


def clean_previous_solution() -> None:
    """Remove code_fixed.py from a previous run so Alan starts fresh."""
    fixed = EXAMPLE_DIR / "code_fixed.py"
    if fixed.exists():
        fixed.unlink()
        print("Removed previous code_fixed.py\n")


def run_tests() -> tuple[bool, str]:
    """Run test_inventory.py and return (passed, output)."""
    result = subprocess.run(
        [sys.executable, str(EXAMPLE_DIR / "test_inventory.py")],
        cwd=str(EXAMPLE_DIR),
        capture_output=True,
        text=True,
        timeout=30,
    )
    output = result.stdout + result.stderr
    passed = result.returncode == 0 and "FAIL" not in output
    return passed, output


def run_with_model(model: str) -> None:
    """Run the auto-fix loop with a real LLM."""
    clean_previous_solution()
    agent = AlanCodeAgent(
        provider="litellm",
        model=model,
        cwd=str(EXAMPLE_DIR),
        permission_mode="yolo",
        max_iterations_per_turn=15,
    )

    prompt = (
        "There are 5 bugs in code_bugged.py. The tests in test_inventory.py expose them.\n"
        "Your task:\n"
        "1. Read code_bugged.py and test_inventory.py\n"
        "2. Create a file called code_fixed.py that is a corrected copy of code_bugged.py\n"
        "3. Do NOT modify code_bugged.py or test_inventory.py\n"
        "4. Fix ALL bugs so that ALL tests pass"
    )

    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"\n{'='*60}")
        print(f"  Attempt {attempt}/{MAX_ATTEMPTS}")
        print(f"{'='*60}\n")

        answer = agent.query(prompt)
        print(f"Alan: {answer[:400]}{'...' if len(answer) > 400 else ''}\n")

        print("Running tests...")
        passed, output = run_tests()
        print(output)

        if passed:
            print(f"\n{'='*60}")
            print(f"  ALL TESTS PASSED on attempt {attempt}!")
            print(f"  Cost: ${agent.cost_usd:.4f}")
            print(f"{'='*60}")

            summary = agent.query(
                "All tests pass now. Give a brief summary of the bugs you found and fixed."
            )
            print(f"\nSummary:\n{summary}")
            return

        prompt = (
            f"The tests still fail. Here is the output:\n\n"
            f"```\n{output}\n```\n\n"
            f"Please read code_fixed.py, find the remaining bugs, and fix them.\n"
            f"Remember: do NOT modify code_bugged.py or test_inventory.py."
        )

    print(f"\nFailed to fix all bugs after {MAX_ATTEMPTS} attempts.")
    print(f"Cost: ${agent.cost_usd:.4f}")


def run_scripted() -> None:
    """Run with scripted provider to demonstrate the loop structure."""
    clean_previous_solution()
    from alancode.providers.scripted_provider import ScriptedProvider, rule, text, tool_call

    bugged = str(EXAMPLE_DIR / "code_bugged.py")
    tests = str(EXAMPLE_DIR / "test_inventory.py")
    fixed = str(EXAMPLE_DIR / "code_fixed.py")

    # Read the actual bugged code so we can script the fixes
    bugged_content = open(bugged).read()

    # Apply all 5 fixes to create the correct content
    fixed_content = bugged_content
    fixed_content = fixed_content.replace(
        "return self.price + self.quantity", "return self.price * self.quantity"
    )
    fixed_content = fixed_content.replace(
        "discount_amount = self.price * percent",
        "discount_amount = self.price * percent / 100",
    )
    fixed_content = fixed_content.replace(
        "product.quantity = quantity", "product.quantity += quantity"
    )
    fixed_content = fixed_content.replace(
        "for product in self.products:", "for product in self.products.values():"
    )
    fixed_content = fixed_content.replace(
        "if product.quantity < threshold:", "if product.quantity <= threshold:"
    )

    provider = ScriptedProvider(rules=[
        # Turn 0: read the bugged code
        rule(turn=0, respond=tool_call("Read", {"file_path": bugged})),
        # Turn 1: read the tests
        rule(turn=1, respond=tool_call("Read", {"file_path": tests})),
        # Turn 2: write the fixed version
        rule(turn=2, respond=tool_call("Write", {"file_path": fixed, "content": fixed_content})),
        # Turn 3: report
        rule(respond=text(
            "I found and fixed 5 bugs in code_bugged.py → code_fixed.py:\n"
            "1. total_value: price + quantity → price * quantity\n"
            "2. apply_discount: percent not / 100\n"
            "3. restock: = → += (add, don't replace)\n"
            "4. total_inventory_value: iterated keys → values\n"
            "5. find_low_stock: < → <= (at or below)"
        )),
    ])

    agent = AlanCodeAgent(provider=provider, cwd=str(EXAMPLE_DIR), permission_mode="yolo")

    print("Attempt 1 (scripted)\n")
    answer = agent.query(
        "Fix the bugs in code_bugged.py → code_fixed.py. Don't modify the tests."
    )
    print(f"Alan: {answer}\n")

    print("Running tests...")
    passed, output = run_tests()
    print(output)

    if passed:
        print("ALL TESTS PASSED!")
    else:
        print("Some tests failed (check the scripted fixes)")

    # Clean up
    fixed_path = EXAMPLE_DIR / "code_fixed.py"
    if fixed_path.exists():
        fixed_path.unlink()
        print("Cleaned up code_fixed.py")


if __name__ == "__main__":
    if "--scripted" in sys.argv:
        run_scripted()
    else:
        model = "openrouter/google/gemini-2.5-flash"
        for i, arg in enumerate(sys.argv):
            if arg == "--model" and i + 1 < len(sys.argv):
                model = sys.argv[i + 1]
        run_with_model(model)
