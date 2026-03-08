"""
Custom encoder for the customer churn predictor model.

Exports:
  ENCODER_CONFIG — EncoderConfig with semantic + metrics layers
  encode_query(query) — converts NL text to a Concept for similarity search
  assess_query(query) — checks query has enough signal (returns ASK if too vague)
  entry_to_record(entry) — converts a JSONL exemplar to an encodable record

Architecture:
  Semantic layer (0.3): identity (customer_id key_part) + context (description BoW, keywords BoW)
    → Provides text-based matching for NL queries about customer health
  Metrics layer (0.7): usage (logins, support_cases, defects, feature_adoption — THERMOMETER)
    → Provides numeric matching for raw customer data records

  Two matching paths, one config:
    - Customer records (metrics, no text) → match on metrics layer
    - NL queries (text, no metrics) → match on semantic layer
    - Exemplars have BOTH → matchable from either direction

  Risk level is the OUTCOME (exemplar metadata), never an encoded attribute.

  Temporal: customer_id is key_part with auto timestamps for daily drift tracking.
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

from intent import extract_keywords, preprocess

# ---------------------------------------------------------------------------
# ENCODER_CONFIG
# ---------------------------------------------------------------------------

ENCODER_CONFIG = EncoderConfig(
    dimension=2000,
    seed=42,
    temporal_source="auto",
    temporal_config=TemporalConfig(signal_type="auto"),
    layers=[
        # --- Semantic layer: text-based matching for NL queries ---
        Layer(
            name="semantic",
            similarity_weight=0.3,
            segments=[
                Segment(
                    name="identity",
                    roles=[
                        Role(
                            name="customer_id",
                            similarity_weight=0.1,
                            key_part=True,
                        ),
                    ],
                ),
                Segment(
                    name="context",
                    roles=[
                        Role(
                            name="description",
                            similarity_weight=1.0,
                            text_encoding="bag_of_words",
                        ),
                        Role(
                            name="keywords",
                            similarity_weight=0.8,
                            text_encoding="bag_of_words",
                        ),
                    ],
                ),
            ],
        ),
        # --- Metrics layer: numeric matching for customer data ---
        Layer(
            name="metrics",
            similarity_weight=0.7,
            segments=[
                Segment(
                    name="usage",
                    roles=[
                        Role(
                            name="logins",
                            similarity_weight=1.0,
                            numeric_config=NumericConfig(
                                bin_width=5.0,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=0.0,
                                max_value=200.0,
                            ),
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
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# ---------------------------------------------------------------------------
# encode_query — NL text → Concept dict
# ---------------------------------------------------------------------------

def encode_query(query: str) -> dict:
    """Convert a raw NL query about churn into a Concept-compatible dict.

    Encodes text into the semantic layer (description + keywords).
    No numeric values — query matching is text-driven.
    Risk/driver labels are outcomes of similarity, not inputs.
    """
    keywords = extract_keywords(query)

    stable_id = int(hashlib.md5(query.encode()).hexdigest()[:8], 16)

    # Use expanded keywords for BOTH description and keywords roles.
    # For NL queries, the raw query text adds noise to description BoW
    # (short common words dominate), while synonym-expanded keywords
    # carry the actual domain signal. Doubling down on keywords via
    # both roles gives consistent, vocabulary-driven matching.
    return {
        "name": f"query_{stable_id:08d}",
        "attributes": {
            "customer_id": "",
            "description": keywords,
            "keywords": keywords,
        },
    }


# ---------------------------------------------------------------------------
# entry_to_record — JSONL exemplar → encodable record + metadata
# ---------------------------------------------------------------------------

def entry_to_record(entry: dict) -> dict:
    """Convert a JSONL exemplar to an encodable record with metadata.

    Attributes encode the behavioral profile (text + metrics).
    Risk/driver labels go to metadata only — returned after matching.
    """
    question = entry.get("question", "")
    customer_id = entry.get("customer_id", "")
    kw_list = entry.get("keywords", [])
    kw_str = " ".join(kw_list) if isinstance(kw_list, list) else str(kw_list)

    slug = re.sub(r"[^a-z0-9]+", "_", question.lower()).strip("_")[:40]

    return {
        "concept_text": question,
        "attributes": {
            "customer_id": customer_id,
            "description": preprocess(question),
            "keywords": kw_str,
            "logins": entry.get("logins", 0),
            "support_cases": entry.get("support_cases", 0),
            "defects": entry.get("defects", 0),
            "feature_adoption": entry.get("feature_adoption", 0),
        },
        "metadata": {
            "record_type": entry.get("record_type", "pattern"),
            "risk_level": entry.get("risk_level", ""),
            "churn_driver": entry.get("churn_driver", ""),
            "recommended_action": entry.get("recommended_action", ""),
            "response": entry.get("response", ""),
            "original_question": question,
            "logins": entry.get("logins", 0),
            "support_cases": entry.get("support_cases", 0),
            "defects": entry.get("defects", 0),
            "feature_adoption": entry.get("feature_adoption", 0),
            **({"gql_id": entry["gql_id"]} if entry.get("gql_id") else {}),
            **({"gql_query": entry["gql_query"]} if entry.get("gql_query") else {}),
        },
    }
