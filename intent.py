"""
Intent extraction for the customer churn predictor model.

Provides keyword extraction and text preprocessing for NL queries
about customer health and churn risk. No risk/driver/band inference —
those are outcomes derived from HDC similarity, not inputs.

Exports:
  extract_keywords(text) → str  — space-separated keyword tokens
  preprocess(text) → str        — cleaned text for BoW encoding
"""

import re


# ---------------------------------------------------------------------------
# Stop words — filtered from keyword extraction
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset({
    "a", "an", "the", "to", "is", "are", "was", "were", "be", "been",
    "do", "does", "did", "doing", "have", "has", "had", "having",
    "how", "what", "which", "who", "whom", "when", "where", "why",
    "i", "me", "my", "we", "our", "you", "your", "it", "its",
    "he", "she", "they", "them", "their", "this", "that", "these",
    "in", "on", "at", "for", "with", "about", "of", "from", "by",
    "can", "will", "would", "could", "should", "may", "might",
    "and", "or", "but", "not", "if", "so", "than",
    "show", "find", "get", "tell", "give", "list",
})


def preprocess(text: str) -> str:
    """Lowercase and strip punctuation for consistent BoW encoding."""
    return re.sub(r"[^\w\s]", " ", text.lower()).strip()


# ---------------------------------------------------------------------------
# Domain synonym expansion — maps broad NL terms to exemplar vocabulary
# ---------------------------------------------------------------------------

_DOMAIN_SYNONYMS: dict[str, list[str]] = {
    # Churn / risk concepts → exemplar keywords they should match
    "churn":       ["churn", "at risk", "inactive", "declining", "cancel", "disengaged"],
    "risk":        ["at risk", "churn", "declining", "frustrated", "watchlist"],
    "cancel":      ["cancel", "churn", "at risk", "leave", "dispute"],
    "leave":       ["churn", "at risk", "disengaged", "cancel"],
    "retain":      ["safe", "retained", "healthy", "engaged"],
    "retention":   ["safe", "retained", "healthy", "engaged"],
    "healthy":     ["healthy", "safe", "retained", "growing", "engaged"],
    "happy":       ["safe", "retained", "satisfied", "engaged"],
    "unhappy":     ["frustrated", "at risk", "churn", "unhappy"],
    "frustrated":  ["frustrated", "at risk", "churn", "support", "tickets"],
    "growing":     ["growing", "expanding", "active", "healthy"],
    "declining":   ["declining", "dropping", "usage drop", "at risk", "churn"],
    "inactive":    ["inactive", "no logins", "zero activity", "churn"],
    "engaged":     ["active", "engaged", "healthy", "growing"],
    "disengaged":  ["disengaged", "inactive", "declining", "churn", "at risk"],
}


def extract_keywords(text: str) -> str:
    """Extract content keywords from NL text, filtering stop words.

    Expands domain-specific terms (churn, risk, cancel, etc.) into
    vocabulary that overlaps with exemplar keywords for BoW matching.

    Returns space-separated keyword tokens.
    """
    cleaned = preprocess(text)
    tokens = [w for w in cleaned.split() if w not in _STOP_WORDS and len(w) > 1]

    # Expand domain synonyms — add related terms for broad query words
    expanded: list[str] = list(tokens)
    for token in tokens:
        synonyms = _DOMAIN_SYNONYMS.get(token)
        if synonyms:
            for syn in synonyms:
                for word in syn.split():
                    if word not in _STOP_WORDS and word not in expanded:
                        expanded.append(word)

    return " ".join(expanded)
