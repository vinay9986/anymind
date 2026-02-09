from __future__ import annotations

import re
from typing import Any


def select_brain_for_question(question: str) -> tuple[str, str, float, dict[str, Any]]:
    """Gold-standard heuristic brain selector.

    Returns (brain, level, score, features) where brain ∈ {"agot","aiot","giot","got"}.

    Mapping:
    - High ambiguity -> got
    - Medium ambiguity -> agot
    - Low ambiguity -> giot
    - None -> aiot
    """
    lvl, score, features = _classify_ambiguity_tier1(question)
    if lvl == "high":
        return "got", lvl, score, features
    if lvl == "medium":
        return "agot", lvl, score, features
    if lvl == "low":
        return "giot", lvl, score, features
    return "aiot", lvl, score, features


def _classify_ambiguity_tier1(question: str) -> tuple[str, float, dict[str, Any]]:
    """Deterministic ambiguity classifier returning (level, score in [0,1], features)."""
    raw = (question or "").strip()
    lowered = raw.lower()

    tokens = re.findall(r"\b[\w']+\b", lowered)
    token_set = set(tokens)
    first_token = tokens[0] if tokens else ""

    yes_no_starters = {
        "is",
        "are",
        "am",
        "was",
        "were",
        "do",
        "does",
        "did",
        "can",
        "could",
        "should",
        "would",
        "will",
        "have",
        "has",
        "had",
        "may",
        "might",
    }
    wh_words = {"who", "when", "where", "what", "which", "why", "how"}

    is_yes_no = first_token in yes_no_starters
    wh_word = first_token if first_token in wh_words else ""

    vague_token_terms = {
        "something",
        "someone",
        "somewhere",
        "somehow",
        "anything",
        "anyone",
        "anywhere",
        "it",
        "this",
        "that",
        "those",
        "these",
        "stuff",
        "things",
        "etc",
    }
    vague_phrase_terms = (
        "in general",
        "generally",
        "typically",
        "kind of",
        "sort of",
    )

    open_token_terms = {
        "design",
        "explore",
        "discover",
        "plan",
        "strategy",
        "architect",
        "trade",
        "pros",
        "cons",
        "recommend",
        "suggest",
        "brainstorm",
    }
    open_phrase_terms = (
        "best way",
        "how should",
        "what do you think",
        "tell me about",
    )

    explanation_token_prefixes = ("analyz",)
    explanation_token_terms = {
        "explain",
        "describe",
        "discuss",
        "evaluate",
        "compare",
        "tradeoffs",
        "steps",
        "guide",
    }
    explanation_phrase_terms = (
        "walk me through",
        "step by step",
    )

    vague_targets = ("scalable", "robust", "reliable", "optimize", "better")

    ranking_terms = (
        "most",
        "least",
        "best",
        "worst",
        "rank",
        "ranking",
        "top",
        "dangerous",
        "safest",
        "strongest",
        "weakest",
        "important",
    )
    criteria_markers = (
        "by ",
        "based on",
        "criteria",
        "use ",
        "using ",
        "rank by",
    )
    criteria_terms = (
        "wind",
        "wind speed",
        "storm surge",
        "surge",
        "rain",
        "rainfall",
        "pressure",
        "category",
        "fatalities",
        "deaths",
        "damage",
        "property damage",
        "impact",
        "affected",
    )

    has_numbers = bool(re.search(r"\b\d+\b", lowered))
    has_timezone_or_units = bool(
        re.search(
            r"\b(utc|gmt|et|est|edt|pt|pst|pdt|ct|cst|cdt|mt|mst|mdt|am|pm|mph|km\/h|kmh)\b",
            lowered,
        )
    )
    has_quotes = any(ch in raw for ch in ('"', "`"))
    has_options = bool(re.search(r"\b\w+\s+or\s+\w+\b", lowered))
    has_urls_or_paths = bool(re.search(r"https?://|/[\w.-]+/|\\[\w.-]+\\", raw))
    proper_nouns = re.findall(r"\b[A-Z][a-z]{2,}\b", raw)
    proper_noun_count = max(0, len(proper_nouns) - 1)

    vague_hits = sum(1 for t in vague_token_terms if t in token_set)
    vague_hits += sum(1 for t in vague_phrase_terms if t in lowered)

    open_hits = sum(1 for t in open_token_terms if t in token_set)
    open_hits += sum(1 for t in open_phrase_terms if t in lowered)

    explain_hits = sum(1 for t in explanation_token_terms if t in token_set)
    explain_hits += sum(
        1
        for pfx in explanation_token_prefixes
        if any(tok.startswith(pfx) for tok in tokens)
    )
    explain_hits += sum(1 for t in explanation_phrase_terms if t in lowered)
    vague_target_hits = sum(1 for t in vague_targets if t in lowered)

    ranking_hit_count = sum(1 for t in ranking_terms if t in token_set)
    has_ranking = ranking_hit_count > 0
    has_explicit_criteria = any(m in lowered for m in criteria_markers) and any(
        t in lowered for t in criteria_terms
    )

    multi_part_separators = lowered.count(",") + lowered.count(";")
    multi_part_connectors = lowered.count(" and ") + lowered.count(" or ")
    multi_part = (multi_part_separators + multi_part_connectors) >= 2

    score = 0.0
    bonuses: dict[str, float] = {}
    penalties: dict[str, float] = {}

    if is_yes_no:
        penalties["yes_no_question"] = 0.25
        score -= penalties["yes_no_question"]
    elif wh_word in {"who", "when", "where"}:
        penalties[f"wh_{wh_word}"] = 0.15
        score -= penalties[f"wh_{wh_word}"]
    elif wh_word == "why":
        bonuses["wh_why"] = 0.35
        score += bonuses["wh_why"]
    elif wh_word == "how":
        bonuses["wh_how"] = 0.25
        score += bonuses["wh_how"]
    elif wh_word in {"what", "which"}:
        bonuses[f"wh_{wh_word}"] = 0.05
        score += bonuses[f"wh_{wh_word}"]

    if vague_hits:
        bonuses["vague_terms"] = 0.15 * min(3, vague_hits)
        score += bonuses["vague_terms"]

    if open_hits:
        bonuses["open_terms"] = 0.18 * min(3, open_hits)
        score += bonuses["open_terms"]

    if explain_hits:
        bonuses["explain_terms"] = 0.08 * min(2, explain_hits)
        score += bonuses["explain_terms"]

    if vague_target_hits:
        bonuses["vague_targets"] = 0.12 * min(2, vague_target_hits)
        score += bonuses["vague_targets"]

    if has_numbers:
        penalties["has_numbers"] = 0.15
        score -= penalties["has_numbers"]

    if has_timezone_or_units:
        penalties["has_units"] = 0.08
        score -= penalties["has_units"]

    if has_quotes:
        penalties["has_quotes"] = 0.12
        score -= penalties["has_quotes"]

    if has_options:
        bonuses["has_options"] = 0.06
        score += bonuses["has_options"]

    if has_urls_or_paths:
        penalties["has_urls_paths"] = 0.1
        score -= penalties["has_urls_paths"]

    if proper_noun_count >= 2:
        penalties["proper_nouns"] = 0.12
        score -= penalties["proper_nouns"]

    if multi_part:
        bonuses["multi_part"] = 0.18
        score += bonuses["multi_part"]

    if has_ranking and not has_explicit_criteria:
        bonuses["ranking_without_criteria"] = 0.2
        score += bonuses["ranking_without_criteria"]

    score = max(0.0, min(1.0, score))

    if score >= 0.75:
        level = "high"
    elif score >= 0.5:
        level = "medium"
    elif score >= 0.25:
        level = "low"
    else:
        level = "none"

    features: dict[str, Any] = {
        "first_token": first_token,
        "wh_word": wh_word,
        "vague_hits": vague_hits,
        "open_hits": open_hits,
        "explain_hits": explain_hits,
        "vague_target_hits": vague_target_hits,
        "has_numbers": has_numbers,
        "has_units": has_timezone_or_units,
        "has_quotes": has_quotes,
        "has_options": has_options,
        "has_urls_paths": has_urls_or_paths,
        "proper_noun_count": proper_noun_count,
        "multi_part": multi_part,
        "has_ranking": has_ranking,
        "has_explicit_criteria": has_explicit_criteria,
        "bonuses": bonuses,
        "penalties": penalties,
    }

    return level, score, features
