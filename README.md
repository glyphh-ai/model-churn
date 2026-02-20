# Customer Churn Predictor

Encodes customer usage metrics into HDC vectors to identify churn risk patterns via similarity matching.

## How It Works

You load your customer data daily (e.g. 100k accounts). Each customer is encoded as a glyph keyed by `customer_id` with an auto-generated timestamp. The model compares each customer's metrics against known churn patterns to surface risk levels, drivers, and recommended actions.

The patterns in `data/patterns.jsonl` define what churn looks like — they're the model's domain expertise. Your customer data is the live input. The HDC similarity engine connects the two.

## Model Structure

```
churn/
├── manifest.yaml          # model identity and metadata
├── config.yaml            # runtime config, auto_load_concepts, test config
├── encoder.py             # EncoderConfig + encode_query + entry_to_record
├── build.py               # package model into .glyphh file
├── tests.py               # test runner entry point
├── data/
│   └── patterns.jsonl     # churn pattern definitions (training data)
├── tests/
│   ├── test-concepts.json # sample customers for testing
│   ├── conftest.py        # shared fixtures
│   ├── test_encoding.py   # config validation, role encoding
│   ├── test_similarity.py # risk ranking correctness
│   ├── test_temporal.py   # temporal snapshot behavior
│   ├── test_queries.py    # NL query attribute inference
│   └── test_metrics.py    # numeric binning edge cases
└── README.md
```

## Metrics Encoded

| Role | Type | Range | Description |
|------|------|-------|-------------|
| customer_id | text (key_part) | — | Stable customer identifier for temporal tracking |
| risk_level | categorical | high/medium/low | Inferred from pattern matching |
| churn_driver | categorical | 6 values | Root cause of churn risk |
| usage_band | categorical | inactive/declining/stable/growing | Activity trend |
| logins | numeric | 0–200, bin_width=10 | Login frequency |
| support_cases | numeric | 0–20, bin_width=1 | Support ticket volume |
| defects | numeric | 0–15, bin_width=1 | Bug encounters |
| feature_adoption | numeric | 0–100%, bin_width=5 | Feature utilization |

## Temporal Behavior

Each daily load creates a new temporal snapshot per customer:
- `customer_id` (key_part) + auto timestamp = unique glyph identity
- Enables TREND, PREDICT, and DRIFT queries across time
- Same customer on Monday vs Tuesday = two distinct glyphs with linked history

## Testing

Run the test suite before deploying:

```bash
# Via CLI
glyphh model test ./churn
glyphh model test ./churn -v
glyphh model test ./churn -k similarity

# Or directly
cd churn/
python tests.py
```

The test suite uses `tests/test-concepts.json` — 10 sample customers with raw metrics only (no risk labels). Tests encode these customers, compare against the training patterns, and verify the model correctly identifies risk from metrics alone.

## Data Format

Training patterns in `data/patterns.jsonl` define what churn looks like:

```json
{
  "question": "customer with zero logins last 30 days",
  "risk_level": "high",
  "churn_driver": "low_usage",
  "usage_band": "inactive",
  "logins": 0,
  "support_cases": 0,
  "defects": 0,
  "feature_adoption": 15,
  "keywords": ["inactive", "no logins"],
  "response": "High churn risk — customer has gone completely inactive.",
  "recommended_action": "Send re-engagement email sequence."
}
```

Customer data uploaded via the listener is raw metrics only — no risk labels:

```json
{
  "customer_id": "acme-corp",
  "logins": 0,
  "support_cases": 0,
  "defects": 0,
  "feature_adoption": 15
}
```

The model figures out the risk by comparing raw customer metrics against the training patterns via similarity search.

## Query Examples

```bash
# Broad similarity search — find customers matching churn signals
glyphh query "customers who stopped logging in"
glyphh query "accounts with excessive support tickets"

# Specific customer prediction
glyphh query "will acme-corp churn in the next 30 days"

# Trend analysis
glyphh query "login trend for acme-corp over 90 days"
```
