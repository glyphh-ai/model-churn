"""
Custom encoder for the customer churn predictor model.

Exports:
  ENCODER_CONFIG — EncoderConfig with numeric binning for usage metrics
  encode_query(query) — converts NL text to a Concept for similarity search
  entry_to_record(entry) — converts a JSONL entry to an encodable record

Uses Glyphh HDC primitives:
  - NumericConfig with THERMOMETER encoding for continuous metrics
  - Temporal layer with auto timestamps for trend tracking
  - Lexicons on roles for NL query matching
  - key_part roles for composite primary keys
"""

import hashlib
import re

from glyphh.core.config import (
    EncoderConfig,
    EncodingStrategy,
    Layer,
    NumericConfig,
    Role,
    Segment,
    TemporalConfig,
)

# ---------------------------------------------------------------------------
# ENCODER_CONFIG — the real Glyphh encoder config
# ---------------------------------------------------------------------------

ENCODER_CONFIG = EncoderConfig(
    dimension=10000,
    seed=42,
    temporal_source="auto",
    temporal_config=TemporalConfig(signal_type="auto"),
    layers=[
        Layer(
            name="semantic",
            similarity_weight=0.6,
            segments=[
                Segment(
                    name="identity",
                    roles=[
                        Role(
                            name="customer_id",
                            similarity_weight=0.1,
                            key_part=True,
                            lexicons=["customer", "account", "client", "customer id"],
                        ),
                        Role(
                            name="risk_level",
                            similarity_weight=0.9,
                            lexicons=["risk", "churn risk", "risk level", "health"],
                        ),
                        Role(
                            name="churn_driver",
                            similarity_weight=0.8,
                            lexicons=["driver", "reason", "cause", "churn driver"],
                        ),
                        Role(
                            name="usage_band",
                            similarity_weight=0.7,
                            lexicons=["usage", "activity", "engagement", "trend"],
                        ),
                    ],
                ),
                Segment(
                    name="context",
                    roles=[
                        Role(
                            name="keywords",
                            similarity_weight=0.5,
                            lexicons=["keywords", "tags", "terms"],
                        ),
                    ],
                ),
            ],
        ),
        Layer(
            name="metrics",
            similarity_weight=0.4,
            segments=[
                Segment(
                    name="usage",
                    roles=[
                        Role(
                            name="logins",
                            similarity_weight=1.0,
                            numeric_config=NumericConfig(
                                bin_width=10.0,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=0.0,
                                max_value=200.0,
                            ),
                            lexicons=["logins", "login count", "sessions", "activity"],
                        ),
                        Role(
                            name="support_cases",
                            similarity_weight=0.9,
                            numeric_config=NumericConfig(
                                bin_width=1.0,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=0.0,
                                max_value=20.0,
                            ),
                            lexicons=["support", "tickets", "cases", "support cases"],
                        ),
                        Role(
                            name="defects",
                            similarity_weight=0.9,
                            numeric_config=NumericConfig(
                                bin_width=1.0,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=0.0,
                                max_value=15.0,
                            ),
                            lexicons=["defects", "bugs", "errors", "crashes"],
                        ),
                        Role(
                            name="feature_adoption",
                            similarity_weight=0.8,
                            numeric_config=NumericConfig(
                                bin_width=5.0,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=0.0,
                                max_value=100.0,
                            ),
                            lexicons=["adoption", "features", "feature usage", "utilization"],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# ---------------------------------------------------------------------------
# NL query helpers
# ---------------------------------------------------------------------------

_USAGE_BANDS = {
    "none": "inactive", "zero": "inactive", "no": "inactive", "inactive": "inactive",
    "dropping": "declining", "declining": "declining", "decreasing": "declining",
    "fewer": "declining", "less": "declining", "reduced": "declining", "low": "declining",
    "steady": "stable", "stable": "stable", "normal": "stable", "average": "stable",
    "increasing": "growing", "growing": "growing", "more": "growing", "high": "growing",
    "active": "growing", "frequent": "growing",
}

_CHURN_DRIVERS = {
    "login": "low_usage", "usage": "low_usage", "logins": "low_usage",
    "session": "low_usage", "activity": "low_usage",
    "support": "support_burden", "ticket": "support_burden", "case": "support_burden",
    "cases": "support_burden", "escalation": "support_burden",
    "defect": "defect_frustration", "bug": "defect_frustration", "crash": "defect_frustration",
    "error": "defect_frustration", "broken": "defect_frustration", "defects": "defect_frustration",
    "feature": "low_adoption", "adoption": "low_adoption", "onboard": "onboarding_stall",
    "billing": "billing_friction", "payment": "billing_friction", "invoice": "billing_friction",
    "price": "billing_friction", "cost": "billing_friction",
}

_RISK_KEYWORDS = {
    "high": ["churn", "cancel", "leaving", "at risk", "critical", "red", "danger",
             "inactive", "zero logins", "no activity", "escalat"],
    "medium": ["declining", "dropping", "fewer", "reduced", "warning", "yellow",
               "slowing", "less active", "some risk"],
    "low": ["stable", "growing", "healthy", "green", "active", "engaged",
            "retained", "happy", "renew"],
}

_STOP_WORDS = {
    "how", "do", "i", "a", "the", "to", "is", "what", "my", "an",
    "can", "does", "it", "in", "on", "for", "with", "me", "about",
    "are", "which", "who", "will", "their", "this", "that", "of",
}


def _infer_risk(words):
    text = " ".join(words)
    for level, triggers in _RISK_KEYWORDS.items():
        if any(t in text for t in triggers):
            return level
    return "medium"


def _infer_driver(words):
    for w in words:
        clean = re.sub(r"[^a-z]", "", w)
        if clean in _CHURN_DRIVERS:
            return _CHURN_DRIVERS[clean]
    return "low_usage"


def _infer_usage_band(words):
    for w in words:
        clean = re.sub(r"[^a-z]", "", w)
        if clean in _USAGE_BANDS:
            return _USAGE_BANDS[clean]
    return "stable"


# ---------------------------------------------------------------------------
# encode_query — NL text → Concept dict
# ---------------------------------------------------------------------------

def encode_query(query: str) -> dict:
    """Convert a raw NL query about churn into a Concept-compatible dict."""
    cleaned = re.sub(r"[^\w\s]", "", query.lower())
    words = cleaned.split()

    risk = _infer_risk(words)
    driver = _infer_driver(words)
    usage_band = _infer_usage_band(words)
    keywords = " ".join(w for w in words if w not in _STOP_WORDS)

    stable_id = int(hashlib.md5(query.encode()).hexdigest()[:8], 16)

    return {
        "name": f"query_{stable_id:08d}",
        "attributes": {
            "customer_id": "",
            "risk_level": risk,
            "churn_driver": driver,
            "usage_band": usage_band,
            "keywords": keywords,
            "logins": 50,
            "support_cases": 2,
            "defects": 1,
            "feature_adoption": 50,
        },
    }


# ---------------------------------------------------------------------------
# entry_to_record — JSONL entry → encodable record + metadata
# ---------------------------------------------------------------------------

def entry_to_record(entry: dict) -> dict:
    """Convert a JSONL entry to an encodable record with metadata."""
    question = entry.get("question", "").lower()
    customer_id = entry.get("customer_id", "")
    risk = entry.get("risk_level", "medium")
    driver = entry.get("churn_driver", "low_usage")
    usage_band = entry.get("usage_band", "stable")
    kw_list = entry.get("keywords", [])
    kw_str = " ".join(kw_list) if isinstance(kw_list, list) else str(kw_list)

    slug = re.sub(r"[^a-z0-9]+", "_", question).strip("_")[:40]

    return {
        "concept_text": question,
        "attributes": {
            "customer_id": customer_id,
            "risk_level": risk,
            "churn_driver": driver,
            "usage_band": usage_band,
            "keywords": kw_str,
            "logins": entry.get("logins", 50),
            "support_cases": entry.get("support_cases", 2),
            "defects": entry.get("defects", 1),
            "feature_adoption": entry.get("feature_adoption", 50),
        },
        "metadata": {
            "response": entry.get("response", ""),
            "risk_level": risk,
            "churn_driver": driver,
            "recommended_action": entry.get("recommended_action", ""),
            "original_question": question,
        },
    }
