"""Shared fixtures for churn model tests."""

import json
import sys
from pathlib import Path

import pytest

# Ensure the model directory is importable
MODEL_DIR = Path(__file__).resolve().parent.parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

TESTS_DIR = Path(__file__).resolve().parent
CONCEPTS_PATH = TESTS_DIR / "test-concepts.json"


@pytest.fixture(scope="session")
def test_customers():
    """Load raw customer data from test-concepts.json."""
    with open(CONCEPTS_PATH) as f:
        return json.load(f)["customers"]


@pytest.fixture(scope="session")
def encoder_config():
    """Import and return the model's ENCODER_CONFIG."""
    from encoder import ENCODER_CONFIG
    return ENCODER_CONFIG


@pytest.fixture(scope="session")
def expected_high_risk(test_customers):
    """Customers we expect the model to classify as high risk."""
    return [c for c in test_customers if c["_expected_risk"] == "high"]


@pytest.fixture(scope="session")
def expected_low_risk(test_customers):
    """Customers we expect the model to classify as low risk."""
    return [c for c in test_customers if c["_expected_risk"] == "low"]


@pytest.fixture(scope="session")
def expected_medium_risk(test_customers):
    """Customers we expect the model to classify as medium risk."""
    return [c for c in test_customers if c["_expected_risk"] == "medium"]
