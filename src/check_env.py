"""
Environment health check for the enterprise-ticket-rag-platform.

This script verifies:
1. Python version
2. Required Python packages
3. Pydantic installation
4. pandas installation
5. python-dotenv installation
6. google-genai installation
7. pytest installation
8. Gemini API key configuration
9. Basic Gemini API connectivity (if an API key is available)

Run from the project root:

    python src/check_env.py
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import Final

from dotenv import load_dotenv


MIN_PYTHON: Final[tuple[int, int]] = (3, 10)

REQUIRED_PACKAGES: Final[dict[str, str]] = {
    "pandas": "pandas",
    "pydantic": "pydantic",
    "dotenv": "python-dotenv",
    "google.genai": "google-genai",
    "pytest": "pytest",
}


def print_status(name: str, success: bool, message: str) -> None:
    """Print a consistent health-check status line."""
    status = "PASS" if success else "FAIL"
    print(f"[{status}] {name}: {message}")


def check_python_version() -> bool:
    """Check whether the Python version meets the minimum requirement."""
    current = sys.version_info[:2]

    if current >= MIN_PYTHON:
        print_status(
            "Python",
            True,
            f"{current[0]}.{current[1]} "
            f"(minimum required: {MIN_PYTHON[0]}.{MIN_PYTHON[1]})",
        )
        return True

    print_status(
        "Python",
        False,
        f"{current[0]}.{current[1]} is too old. "
        f"Minimum required: {MIN_PYTHON[0]}.{MIN_PYTHON[1]}",
    )
    return False


def check_packages() -> bool:
    """Check whether all required Python packages can be imported."""
    all_installed = True

    for import_name, package_name in REQUIRED_PACKAGES.items():
        try:
            module = importlib.import_module(import_name)

            version = getattr(module, "__version__", "installed")

            print_status(
                package_name,
                True,
                f"{version}",
            )

        except ImportError:
            print_status(
                package_name,
                False,
                "Package is not installed or cannot be imported.",
            )
            all_installed = False

    return all_installed


def check_gemini_api_key() -> bool:
    """Check whether GEMINI_API_KEY exists in the environment."""
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print_status(
            "Gemini API key",
            False,
            "GEMINI_API_KEY is not set. "
            "Create a .env file before running Gemini-based tasks.",
        )
        return False

    if api_key == "your_gemini_api_key_here":
        print_status(
            "Gemini API key",
            False,
            "Placeholder key detected. Replace it with your actual API key.",
        )
        return False

    print_status(
        "Gemini API key",
        True,
        "API key is configured.",
    )

    return True


def check_gemini_connection() -> bool:
    """
    Perform a minimal Gemini API connectivity test.

    This only runs when GEMINI_API_KEY is configured.
    """
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key or api_key == "your_gemini_api_key_here":
        print_status(
            "Gemini API connection",
            False,
            "Skipped because GEMINI_API_KEY is not configured.",
        )
        return False

    try:
        from google import genai

        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents="Reply with exactly: ZYCUS_ENV_OK",
        )

        text = (response.text or "").strip()

        if text:
            print_status(
                "Gemini API connection",
                True,
                f"Received response: {text[:100]}",
            )
            return True

        print_status(
            "Gemini API connection",
            False,
            "API request succeeded but returned an empty response.",
        )
        return False

    except Exception as exc:
        print_status(
            "Gemini API connection",
            False,
            f"{type(exc).__name__}: {exc}",
        )
        return False


def main() -> int:
    """Run all environment checks."""
    print("=" * 60)
    print("ZYCUS AI ASSESSMENT - ENVIRONMENT HEALTH CHECK")
    print("=" * 60)
    print()

    python_ok = check_python_version()
    print()

    packages_ok = check_packages()
    print()

    api_key_ok = check_gemini_api_key()
    print()

    if api_key_ok:
        gemini_ok = check_gemini_connection()
    else:
        gemini_ok = False
        print_status(
            "Gemini API connection",
            False,
            "Skipped because the API key check failed.",
        )

    print()
    print("=" * 60)

    if python_ok and packages_ok and api_key_ok and gemini_ok:
        print("RESULT: ENVIRONMENT READY")
        print("=" * 60)
        return 0

    print("RESULT: ENVIRONMENT NEEDS ATTENTION")
    print("=" * 60)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())