# Customer Churn Predictor

Encodes customer usage metrics into HDC vectors to identify churn risk patterns via similarity matching. Supports natural language queries with domain synonym expansion, morphological normalization, and two-stage exemplar-to-customer matching.

## How It Works

You load your customer data daily (e.g. 100k accounts). Each customer is encoded as a glyph keyed by `customer_id` with an auto-generated timestamp. The model compares each customer's metrics against known churn patterns to surface risk levels, drivers, and recommended actions.

**Two-stage query pipeline:**

```
NL query ("what customers are likely to churn?")
  ↓
intent.py → keyword extraction + domain synonym expansion
  ↓
HDC encode → similarity search against exemplars (semantic layer)
  ↓
Gap analysis + confidence threshold → DONE or ASK
  ↓
If DONE: Stage 2 GQL → find customers matching the exemplar's metrics
  ↓
FactTree response: matched exemplar context + customer data results
```

The exemplars in `data/exemplars.jsonl` define what churn looks like — they're the model's domain expertise. Your customer data is the live input. The HDC similarity engine connects the two.

## Model Structure

```
churn/
├── manifest.yaml          # model identity and metadata
├── config.yaml            # runtime config, thresholds, GQL template
├── encoder.py             # EncoderConfig + encode_query + entry_to_record
├── intent.py              # domain keyword extraction + synonym expansion
├── build.py               # package model into .glyphh file
├── seed_demo.py           # loads demo customers into the running model
├── tests.py               # test runner entry point
├── data/
│   └── exemplars.jsonl    # churn exemplar definitions — auto-loaded at startup
├── demo/
│   └── customers.jsonl    # 25 synthetic customer records for demo/testing
├── tests/
│   ├── test-concepts.json # sample customers for testing
│   ├── conftest.py        # shared fixtures
│   ├── test_encoding.py   # config validation, role encoding
│   ├── test_similarity.py # risk ranking correctness
│   ├── test_temporal.py   # temporal snapshot behavior
│   ├── test_queries.py    # NL query attribute inference
│   ├── test_nl_queries.py # NL→exemplar matching (churn/cancel/healthy)
│   └── test_metrics.py    # numeric binning edge cases
└── README.md
```

**Two-tier data model:**
- `data/exemplars.jsonl` — the model's *domain expertise*. Auto-loaded at startup. Defines what different risk profiles look like in the HDC similarity space.
- Customer records — *runtime data*. Ingested via the listener API. Real (or demo) customer accounts that queries match against.

## NL Query Pipeline

**intent.py** handles domain-specific query preprocessing:

- **Keyword extraction** — filters stop words, extracts content tokens
- **Domain synonym expansion** — maps broad terms to exemplar vocabulary:
  - "churn" → churn, at risk, inactive, declining, cancel, disengaged
  - "healthy" → healthy, safe, retained, growing, engaged
  - "frustrated" → frustrated, at risk, churn, support, tickets

**SDK-level morphological normalization** (automatic, no model code needed):
- Plurals: "days" → "day", "customers" → "customer", "tickets" → "ticket"
- Past tense: "walked" → "walk", "checked" → "check"
- Gerunds: "running" → "run", "building" → "build"

**Confidence gates** (configured in `config.yaml`):
- `similarity.threshold: 0.45` — below this, return ASK instead of DONE
- `disambiguation.min_gap: 0.015` — if top scores cluster, return ASK for disambiguation

**Two-stage execution:**
When an exemplar matches confidently, the runtime executes a Stage 2 GQL query to find customer records with similar metric profiles:
```yaml
gql_query_default: 'FIND SIMILAR TO glyph("{matched_id}") AT LAYER metrics LIMIT 10 THRESHOLD 0.5'
```

## Encoded Roles

**Semantic layer** (0.3 weight) — text matching for NL queries:

| Role | Type | Description |
|------|------|-------------|
| customer_id | text (key_part) | Stable customer identifier for temporal tracking |
| description | bag_of_words | Behavioral description text |
| keywords | bag_of_words | Extracted content keywords |

**Metrics layer** (0.7 weight) — numeric matching for customer data:

| Role | Type | Range | Description |
|------|------|-------|-------------|
| logins | numeric (thermometer) | 0–200, bin_width=5 | Login frequency |
| support_cases | numeric (thermometer) | 0–20, bin_width=1 | Support ticket volume |
| defects | numeric (thermometer) | 0–15, bin_width=1 | Bug encounters |
| feature_adoption | numeric (thermometer) | 0–100%, bin_width=5 | Feature utilization |

**Metadata** (returned after matching, never encoded):

| Field | Description |
|-------|-------------|
| risk_level | high/medium/low — derived from matched exemplar |
| churn_driver | Root cause of churn risk |
| recommended_action | Suggested intervention |

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

The test suite includes:
- **test_encoding.py** — config validation, layer structure, role encoding
- **test_metrics.py** — numeric binning edge cases, boundary values
- **test_similarity.py** — risk ranking correctness (high-risk ↔ high-risk)
- **test_temporal.py** — temporal snapshot and key_part behavior
- **test_queries.py** — NL query keyword extraction and attribute inference
- **test_nl_queries.py** — end-to-end NL→exemplar matching (churn queries → high risk, healthy queries → low risk)

## Data Format

Exemplars in `data/exemplars.jsonl` define what churn looks like:

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
  "keywords": ["inactive", "no logins", "zero activity", "churn", "at risk", "disengaged"],
  "response": "High churn risk — customer has gone completely inactive.",
  "recommended_action": "Send re-engagement email sequence and schedule CSM outreach within 48 hours."
}
```

Customer data uploaded via the listener is raw metrics only — no risk labels, no pre-computed drivers:

```json
{
  "customer_id": "acme-corp",
  "concept_text": "acme-corp — 0 logins, 0 support cases, 15% adoption",
  "logins": 0,
  "support_cases": 0,
  "defects": 0,
  "feature_adoption": 15
}
```

The model infers risk, churn driver, and recommended actions by comparing raw customer metrics against the exemplars via similarity search.

## Running the Demo

Exemplars load automatically at startup. Customer records are ingested separately via the listener API — the same path real customer data would take.

```bash
# 1. Start the model
glyphh dev . -d

# 2. Seed 25 demo customers
python seed_demo.py

# 3. Chat
glyphh chat
```

### Demo queries to try

```
what customers are likely to churn in the next 30 days?
show me accounts with no logins
who has excessive support tickets?
which customers have low feature adoption?
find accounts approaching renewal that are at risk
show me healthy growing accounts
customers with defect frustration
who is completely inactive?
find accounts with billing issues
which customers should I prioritize this week?
```

### Loading real customer data

POST records to the listener API. Send raw metrics only — the model infers risk level, churn driver, and recommended actions by matching against the exemplars in `data/exemplars.jsonl`.

```bash
curl -X POST http://localhost:8002/local-dev-org/churn/listener \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      {
        "customer_id": "acme-corp",
        "concept_text": "acme-corp — 0 logins, 0 support cases, 15% adoption",
        "logins": 0,
        "support_cases": 0,
        "defects": 0,
        "feature_adoption": 15
      }
    ]
  }'
```

`concept_text` is the display label shown in search results. The model determines risk, driver, and recommendations from the metrics alone.

## Query Examples

```bash
# Broad similarity search — find customers matching churn signals
glyphh query "customers who stopped logging in"
glyphh query "accounts with excessive support tickets"

# Specific customer lookup
glyphh query "what is acme-corp's churn risk?"

# Trend analysis
glyphh query "login trend for acme-corp over 90 days"
```
