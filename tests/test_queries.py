"""Test that encode_query produces correct attributes from NL text."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from encoder import encode_query


def test_high_risk_query_infers_high():
    """Queries with high-risk signals should infer risk_level=high."""
    result = encode_query("customer is completely inactive with zero logins")
    assert result["attributes"]["risk_level"] == "high"


def test_low_risk_query_infers_low():
    """Queries with healthy signals should infer risk_level=low."""
    result = encode_query("customer is active and engaged with growing usage")
    assert result["attributes"]["risk_level"] == "low"


def test_medium_risk_query_infers_medium():
    """Queries with moderate signals should infer risk_level=medium."""
    result = encode_query("customer usage is declining slightly")
    assert result["attributes"]["risk_level"] == "medium"


def test_driver_inference_support():
    """Mentions of support/tickets should map to support_burden driver."""
    result = encode_query("customer has many support tickets")
    assert result["attributes"]["churn_driver"] == "support_burden"


def test_driver_inference_defects():
    """Mentions of bugs/defects should map to defect_frustration driver."""
    result = encode_query("customer keeps hitting the same bug")
    assert result["attributes"]["churn_driver"] == "defect_frustration"


def test_driver_inference_billing():
    """Mentions of billing/invoice should map to billing_friction driver."""
    result = encode_query("customer disputed their invoice")
    assert result["attributes"]["churn_driver"] == "billing_friction"


def test_usage_band_inactive():
    """Inactive signals should map to inactive usage_band."""
    result = encode_query("customer has no activity at all")
    assert result["attributes"]["usage_band"] == "inactive"


def test_usage_band_growing():
    """Growing signals should map to growing usage_band."""
    result = encode_query("customer logins are increasing rapidly")
    assert result["attributes"]["usage_band"] == "growing"


def test_query_has_customer_id_empty():
    """Queries are not specific customers, so customer_id should be empty."""
    result = encode_query("which customers are likely to churn")
    assert result["attributes"]["customer_id"] == ""


def test_query_has_stable_name():
    """Same query text should produce the same concept name."""
    q = "find at-risk customers"
    r1 = encode_query(q)
    r2 = encode_query(q)
    assert r1["name"] == r2["name"]


def test_keywords_exclude_stop_words():
    """Keywords should filter out common stop words."""
    result = encode_query("how do I find the customers at risk")
    kw = result["attributes"]["keywords"]
    assert "how" not in kw.split()
    assert "do" not in kw.split()
    assert "the" not in kw.split()
