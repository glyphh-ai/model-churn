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
    "and", "or", "but", "not", "if", "so", "than", "too", "very",
    "show", "find", "get", "tell", "give", "list", "see", "look",
    "us", "me", "there", "here", "just", "also", "been", "being",
    "some", "any", "all", "most", "many", "much", "more", "less",
    "product", "company", "platform", "service", "tool", "software",
    "customers", "customer", "accounts", "account", "clients", "client",
    "users", "user",
})


def preprocess(text: str) -> str:
    """Lowercase and strip punctuation for consistent BoW encoding."""
    return re.sub(r"[^\w\s]", " ", text.lower()).strip()


# ---------------------------------------------------------------------------
# Light stemming — normalize common suffixes before synonym lookup
# ---------------------------------------------------------------------------

# Ordered longest-first so we strip the longest matching suffix
_STEM_SUFFIXES = [
    "ation", "tion", "ment", "ness", "able", "ible",
    "ing", "ers", "ies", "ous", "ive",
    "ed", "es", "er", "ly",
    "s",
]

# Exceptions: words that should NOT be stemmed (stem would be wrong)
_STEM_EXCEPTIONS = frozenset({
    "this", "has", "was", "does", "goes", "is", "us", "yes", "no",
    "less", "plus", "bus", "gas", "his", "its",
    "cases", "issues",  # keep these as-is, they have synonym entries
})


def _stem(word: str) -> str:
    """Strip the longest common suffix if the remaining stem is >= 3 chars."""
    if word in _STEM_EXCEPTIONS or len(word) <= 3:
        return word
    for suffix in _STEM_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


# ---------------------------------------------------------------------------
# Phrase normalization — compound terms → single tokens
# ---------------------------------------------------------------------------

_PHRASE_MAP = {
    "at risk":           "at_risk",
    "feature adoption":  "feature_adoption",
    "low usage":         "low_usage",
    "low adoption":      "low_adoption",
    "support tickets":   "support_tickets",
    "support cases":     "support_cases",
    "support burden":    "support_burden",
    "power user":        "power_user",
    "power users":       "power_user",
    "no logins":         "no_logins",
    "zero logins":       "no_logins",
    "likely to":         "likely_to",
    "about to":          "about_to",
    "going to":          "going_to",
    "churn risk":        "churn_risk",
    "high risk":         "high_risk",
    "red flag":          "red_flag",
    "went dark":         "went_dark",
    "gone dark":         "went_dark",
    "dropped off":       "dropped_off",
    "fell off":          "dropped_off",
}


def _apply_phrases(text: str) -> str:
    """Replace known multi-word phrases with underscore-joined tokens.

    Matches longest phrases first so "power users" is replaced before
    "power user" (which would leave a trailing 's').
    """
    for phrase, token in sorted(_PHRASE_MAP.items(), key=lambda x: -len(x[0])):
        text = text.replace(phrase, token)
    return text


# ---------------------------------------------------------------------------
# Domain synonym expansion — maps broad NL terms to exemplar vocabulary
# ---------------------------------------------------------------------------

_DOMAIN_SYNONYMS: dict[str, list[str]] = {
    # ── Churn / exit signals ──
    "churn":        ["churn", "at_risk", "inactive", "declining", "cancel", "disengaged", "likely to leave"],
    "churning":     ["churn", "at_risk", "inactive", "declining", "cancel", "disengaged"],
    "attrition":    ["churn", "at_risk", "declining", "cancel", "disengaged", "losing"],
    "turnover":     ["churn", "at_risk", "declining", "cancel", "losing"],
    "defection":    ["churn", "at_risk", "cancel", "disengaged", "abandon"],
    "abandon":      ["churn", "at_risk", "cancel", "disengaged", "inactive", "abandon"],
    "quit":         ["churn", "at_risk", "cancel", "disengaged", "abandon"],
    "gone":         ["churn", "inactive", "disengaged", "zero activity", "abandon"],
    "disappear":    ["churn", "inactive", "disengaged", "zero activity", "no logins"],
    "loss":         ["churn", "at_risk", "declining", "cancel", "losing"],
    "losing":       ["churn", "at_risk", "declining", "cancel", "losing"],
    "lost":         ["churn", "inactive", "disengaged", "cancel"],

    # ── Risk terms ──
    "risk":         ["at_risk", "churn", "declining", "frustrated", "watchlist"],
    "risky":        ["at_risk", "churn", "declining", "frustrated", "watchlist"],
    "danger":       ["at_risk", "churn", "declining", "frustrated", "watchlist"],
    "concern":      ["at_risk", "watchlist", "moderate risk", "early warning"],
    "warning":      ["at_risk", "watchlist", "early warning", "declining"],
    "alert":        ["at_risk", "watchlist", "early warning", "churn"],
    "flag":         ["at_risk", "watchlist", "early warning", "churn"],
    "trouble":      ["at_risk", "frustrated", "churn", "struggling"],
    "problem":      ["at_risk", "frustrated", "churn", "issue", "struggling"],
    "vulnerable":   ["at_risk", "churn", "declining", "watchlist"],
    "jeopardy":     ["at_risk", "churn", "declining", "cancel"],

    # ── Cancel / billing ──
    "cancel":       ["cancel", "churn", "at_risk", "leave", "dispute"],
    "unsubscribe":  ["cancel", "churn", "at_risk", "leave"],
    "downgrade":    ["cancel", "churn", "at_risk", "declining", "billing"],
    "dispute":      ["dispute", "billing", "cancel", "churn", "at_risk", "refund"],
    "billing":      ["billing", "invoice", "dispute", "payment", "churn", "at_risk"],
    "invoice":      ["billing", "invoice", "dispute", "payment", "churn"],
    "payment":      ["billing", "payment", "dispute", "invoice"],
    "overcharge":   ["billing", "dispute", "refund", "churn", "at_risk"],
    "refund":       ["refund", "billing", "dispute", "cancel", "churn", "at_risk"],
    "price":        ["billing", "pricing", "dispute", "cancel", "at_risk"],
    "pricing":      ["billing", "pricing", "dispute", "cancel", "at_risk"],
    "expensive":    ["billing", "pricing", "cancel", "at_risk"],
    "cost":         ["billing", "pricing", "cancel", "at_risk"],

    # ── Retention / health ──
    "retain":       ["safe", "retained", "healthy", "engaged"],
    "retention":    ["safe", "retained", "healthy", "engaged"],
    "loyal":        ["safe", "retained", "healthy", "engaged", "committed", "champion"],
    "loyalty":      ["safe", "retained", "healthy", "engaged", "committed"],
    "satisfied":    ["safe", "retained", "satisfied", "healthy", "engaged"],
    "satisfaction": ["safe", "retained", "satisfied", "healthy"],
    "happy":        ["safe", "retained", "satisfied", "engaged", "healthy"],
    "content":      ["safe", "retained", "satisfied", "healthy", "stable"],
    "stable":       ["stable", "safe", "retained", "healthy", "steady"],
    "steady":       ["stable", "safe", "retained", "healthy"],
    "reliable":     ["safe", "retained", "healthy", "stable", "consistent"],
    "consistent":   ["safe", "retained", "healthy", "stable"],
    "committed":    ["safe", "retained", "committed", "renewed", "engaged"],
    "best":         ["safe", "retained", "champion", "power user", "engaged", "healthy"],
    "strong":       ["safe", "retained", "healthy", "engaged", "growing"],

    # ── Engagement ──
    "engaged":      ["active", "engaged", "healthy", "growing"],
    "active":       ["active", "engaged", "healthy", "growing"],
    "usage":        ["active", "engaged", "logins", "usage"],
    "login":        ["active", "logins", "engaged", "usage"],
    "frequent":     ["active", "engaged", "daily", "power user"],
    "regular":      ["active", "engaged", "healthy", "stable"],
    "daily":        ["active", "engaged", "daily", "power user"],
    "weekly":       ["active", "engaged", "regular"],
    "power":        ["power user", "champion", "full adoption", "engaged", "safe"],
    "champion":     ["champion", "power user", "advocate", "safe", "retained"],
    "advocate":     ["champion", "advocate", "power user", "safe", "retained"],
    "promoter":     ["champion", "advocate", "safe", "retained"],

    # ── Disengagement ──
    "disengaged":   ["disengaged", "inactive", "declining", "churn", "at_risk"],
    "inactive":     ["inactive", "no logins", "zero activity", "churn", "disengaged"],
    "dormant":      ["inactive", "no logins", "zero activity", "churn", "disengaged", "ghost"],
    "silent":       ["inactive", "disengaged", "zero activity", "ghost"],
    "ghost":        ["inactive", "no logins", "zero activity", "churn", "disengaged", "abandon"],
    "dark":         ["inactive", "no logins", "zero activity", "churn", "disengaged"],
    "unresponsive": ["inactive", "disengaged", "zero activity", "churn"],
    "absent":       ["inactive", "no logins", "zero activity", "disengaged"],
    "missing":      ["inactive", "no logins", "disengaged", "churn"],
    "stopped":      ["inactive", "declining", "disengaged", "churn", "abandon"],
    "stalled":      ["stalled", "onboarding", "incomplete", "at_risk", "churn"],
    "frozen":       ["inactive", "stalled", "disengaged", "churn"],
    "idle":         ["inactive", "no logins", "disengaged", "zero activity"],
    "dead":         ["inactive", "no logins", "zero activity", "churn", "abandon"],

    # ── Adoption / onboarding ──
    "adoption":     ["adoption", "feature adoption"],
    "onboarding":   ["onboarding", "setup", "stalled", "incomplete"],
    "onboard":      ["onboarding", "setup", "stalled", "incomplete"],
    "setup":        ["onboarding", "setup", "incomplete", "stalled"],
    "configure":    ["onboarding", "setup", "incomplete"],
    "implement":    ["onboarding", "setup", "incomplete", "integrate"],
    "integrate":    ["onboarding", "setup", "integrate"],
    "deploy":       ["onboarding", "setup", "deploy"],
    "utilize":      ["adoption", "underutilized", "feature adoption"],
    "underutilize": ["adoption", "underutilized", "low value", "one feature", "switching"],
    "unused":       ["underutilized", "low adoption", "one feature", "at_risk"],
    "untouched":    ["underutilized", "low adoption", "one feature", "at_risk"],

    # ── Support / frustration ──
    "support":      ["support", "tickets", "frustrated", "churn", "at_risk"],
    "ticket":       ["support", "tickets", "frustrated", "many cases"],
    "tickets":      ["support", "tickets", "frustrated", "many cases"],
    "case":         ["support", "tickets", "cases", "frustrated"],
    "cases":        ["support", "tickets", "cases", "frustrated"],
    "escalation":   ["support", "tickets", "frustrated", "escalated", "at_risk"],
    "escalate":     ["support", "tickets", "frustrated", "escalated", "at_risk"],
    "complaint":    ["support", "frustrated", "unhappy", "at_risk"],
    "complain":     ["support", "frustrated", "unhappy", "at_risk"],
    "frustrated":   ["frustrated", "at_risk", "churn", "support", "tickets", "unhappy"],
    "frustration":  ["frustrated", "at_risk", "churn", "support", "unhappy"],
    "angry":        ["frustrated", "at_risk", "churn", "unhappy", "cancel"],
    "unhappy":      ["frustrated", "at_risk", "churn", "unhappy", "dissatisfied"],
    "dissatisfied": ["frustrated", "at_risk", "churn", "unhappy", "dissatisfied"],
    "struggling":   ["frustrated", "at_risk", "support", "struggling", "friction"],
    "friction":     ["frustrated", "friction", "support", "at_risk"],
    "pain":         ["frustrated", "friction", "support", "at_risk"],
    "issue":        ["support", "frustrated", "friction", "at_risk"],
    "issues":       ["support", "frustrated", "friction", "at_risk"],
    "affected":     ["bug", "defect", "quality", "frustrated", "at_risk"],
    "dealing":      ["support", "frustrated", "friction", "at_risk"],

    # ── Quality / defects ──
    "bug":          ["bug", "defect", "broken", "quality", "frustrated", "at_risk"],
    "bugs":         ["bug", "defect", "broken", "quality", "frustrated", "at_risk"],
    "defect":       ["bug", "defect", "recurring", "broken", "quality", "frustrated"],
    "defects":      ["bug", "defect", "recurring", "broken", "quality", "frustrated"],
    "broken":       ["bug", "defect", "broken", "quality", "frustrated", "at_risk"],
    "crash":        ["bug", "defect", "broken", "quality", "frustrated"],
    "error":        ["bug", "defect", "broken", "quality", "frustrated"],
    "failure":      ["bug", "defect", "broken", "quality", "at_risk"],
    "failing":      ["bug", "defect", "broken", "quality", "at_risk"],
    "glitch":       ["bug", "defect", "broken", "quality"],
    "regression":   ["bug", "defect", "recurring", "quality"],
    "quality":      ["bug", "defect", "quality", "frustrated", "at_risk"],
    "unstable":     ["bug", "defect", "broken", "quality", "unreliable"],
    "unreliable":   ["bug", "defect", "broken", "quality", "unreliable"],

    # ── Growth / expansion ──
    "growing":      ["growing", "expanding", "active", "healthy", "strong"],
    "growth":       ["growing", "expanding", "active", "healthy"],
    "expanding":    ["growing", "expanding", "active", "healthy"],
    "expansion":    ["growing", "expanding", "added seats", "healthy"],
    "scaling":      ["growing", "expanding", "active"],
    "upsell":       ["growing", "expanding", "added seats", "opportunity"],
    "upgrade":      ["growing", "expanding", "committed"],
    "adding":       ["growing", "expanding", "added seats"],
    "seats":        ["growing", "expanding", "added seats"],
    "renewal":      ["renewed", "committed", "safe", "retained"],
    "renew":        ["renewed", "committed", "safe", "retained"],
    "extend":       ["renewed", "committed", "growing"],

    # ── Decline ──
    "declining":    ["declining", "dropping", "usage drop", "at_risk", "churn"],
    "decline":      ["declining", "dropping", "usage drop", "at_risk", "churn"],
    "dropping":     ["declining", "dropping", "usage drop", "at_risk"],
    "drop":         ["declining", "dropping", "usage drop", "at_risk"],
    "decreasing":   ["declining", "dropping", "usage drop", "at_risk"],
    "falling":      ["declining", "dropping", "usage drop", "at_risk"],
    "shrinking":    ["declining", "dropping", "at_risk"],
    "reduced":      ["declining", "dropping", "at_risk"],
    "slipping":     ["declining", "dropping", "at_risk", "early warning"],
    "slowing":      ["declining", "dropping", "early warning", "watchlist"],
    "waning":       ["declining", "dropping", "at_risk", "disengaged"],
    "fading":       ["declining", "dropping", "at_risk", "disengaged"],

    # ── Urgency / likelihood ──
    "likely":       ["churn", "at_risk", "likely to leave", "cancel"],
    "imminent":     ["churn", "at_risk", "cancel", "urgent"],
    "soon":         ["churn", "at_risk", "watchlist"],
    "upcoming":     ["renewal", "deadline", "watchlist"],
    "expiring":     ["renewal", "deadline", "at_risk"],
    "deadline":     ["renewal", "deadline", "pressure", "at_risk"],
    "urgent":       ["churn", "at_risk", "cancel", "escalated"],
    "immediate":    ["churn", "at_risk", "cancel", "urgent"],

    # ── Compound phrase tokens (from _PHRASE_MAP) ──
    "at_risk":          ["at_risk", "churn", "declining", "frustrated", "watchlist", "cancel"],
    "likely_to":        ["churn", "at_risk", "likely to leave", "cancel", "inactive", "disengaged"],
    "about_to":         ["churn", "at_risk", "cancel", "imminent"],
    "going_to":         ["churn", "at_risk", "cancel"],
    "churn_risk":       ["churn", "at_risk", "declining", "cancel", "watchlist"],
    "high_risk":        ["churn", "at_risk", "cancel", "frustrated", "declining"],
    "no_logins":        ["inactive", "no logins", "zero activity", "churn", "disengaged"],
    "low_usage":        ["inactive", "declining", "disengaged", "churn", "at_risk"],
    "low_adoption":     ["underutilized", "one feature", "low value", "switching", "at_risk"],
    "feature_adoption": ["adoption", "underutilized", "low value", "one feature"],
    "support_tickets":  ["support", "tickets", "frustrated", "many cases", "at_risk"],
    "support_cases":    ["support", "tickets", "frustrated", "many cases", "at_risk"],
    "support_burden":   ["support", "tickets", "frustrated", "overwhelmed", "at_risk"],
    "power_user":       ["power user", "champion", "full adoption", "engaged", "safe"],
    "went_dark":        ["inactive", "no logins", "zero activity", "ghost", "churn"],
    "dropped_off":      ["declining", "dropping", "usage drop", "at_risk", "churn"],
    "red_flag":         ["at_risk", "churn", "warning", "watchlist"],

}


def extract_keywords(text: str) -> str:
    """Extract content keywords from NL text, filtering stop words.

    Applies phrase normalization, light stemming, and domain synonym
    expansion to produce a keyword string that overlaps with exemplar
    keywords for BoW matching.

    Returns space-separated keyword tokens.
    """
    cleaned = preprocess(text)

    # Phase 1: normalize compound phrases
    cleaned = _apply_phrases(cleaned)

    # Phase 2: tokenize and filter
    tokens = [w for w in cleaned.split() if w not in _STOP_WORDS and len(w) > 1]

    # Phase 3: expand domain synonyms (try raw token, then stemmed)
    expanded: list[str] = list(tokens)
    for token in tokens:
        synonyms = _DOMAIN_SYNONYMS.get(token)
        if not synonyms:
            stemmed = _stem(token)
            synonyms = _DOMAIN_SYNONYMS.get(stemmed)
        if synonyms:
            for syn in synonyms:
                for word in syn.split():
                    if word not in _STOP_WORDS and word not in expanded:
                        expanded.append(word)

    return " ".join(expanded)
