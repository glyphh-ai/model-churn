"""Test that encode_query produces correct attributes from NL text.

encode_query extracts text for semantic matching — description and keywords.
Risk/driver/band labels are outcomes of similarity, not inputs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from encoder import encode_query


def test_query_has_description():
    """Queries should have cleaned text as description for BoW matching."""
    result = encode_query("customer is completely inactive with zero logins")
    desc = result["attributes"]["description"]
    assert "inactive" in desc
    assert "logins" in desc


def test_query_has_keywords():
    """Queries should extract content keywords."""
    result = encode_query("customer has many support tickets and defects")
    kw = result["attributes"]["keywords"]
    assert "support" in kw
    assert "tickets" in kw
    assert "defects" in kw


def test_keywords_exclude_stop_words():
    """Keywords should filter out common stop words."""
    result = encode_query("how do I find the customers at risk")
    kw = result["attributes"]["keywords"]
    assert "how" not in kw.split()
    assert "do" not in kw.split()
    assert "the" not in kw.split()


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


def test_query_has_no_numeric_values():
    """Queries should not contain numeric metric values — matching is text-driven."""
    result = encode_query("customers likely to churn")
    attrs = result["attributes"]
    assert "logins" not in attrs
    assert "support_cases" not in attrs
    assert "defects" not in attrs
    assert "feature_adoption" not in attrs


def test_query_has_no_risk_labels():
    """Queries should not infer risk/driver/band — those are outcomes."""
    result = encode_query("show me at-risk customers with support issues")
    attrs = result["attributes"]
    assert "risk_level" not in attrs
    assert "churn_driver" not in attrs
    assert "usage_band" not in attrs


def test_description_is_lowercased():
    """Description should be lowercased for consistent BoW encoding."""
    result = encode_query("Customer Has HIGH Support Cases")
    desc = result["attributes"]["description"]
    assert desc == desc.lower()
