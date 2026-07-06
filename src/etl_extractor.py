"""
src/etl_extractor.py — Pluggable ETL extraction backends for maintenance work order text.

Three extraction strategies:
    1. rule_based  — regex + keyword lookup (free, <1 ms/record, ~60–70% F1 on root cause)
    2. llm         — GPT-4o-mini with Pydantic structured output (~$0.10/1k records, ~90%+ F1)
    3. hybrid      — rule_based first; route low-confidence records to llm (~$0.02/1k, ~94% F1)

Each backend accepts raw work-order text and returns a WorkOrderFields Pydantic model.

Usage:
    extractor = ETLExtractor(mode="hybrid")
    fields = extractor.extract("Replaced mechanical seal on P-104 due to bearing wear.")

Requires .env with OPENAI_API_KEY for llm and hybrid modes.
"""

from __future__ import annotations

import os
import re
import time
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# ─── Output schema ────────────────────────────────────────────────────────────

class WorkOrderFields(BaseModel):
    equipment_tag:    Optional[str] = Field(None, description="Equipment identifier e.g. P-104")
    failure_mode:     Optional[str] = Field(None, description="Primary failure mode")
    parts_replaced:   Optional[str] = Field(None, description="Parts replaced or consumed")
    root_cause:       Optional[str] = Field(None, description="Root cause of failure")
    failure_category: Optional[str] = Field(None, description="One of: mechanical_failure, electrical_failure, hydraulic_failure, instrumentation_failure, preventive_maintenance, operator_damage")
    confidence:       float         = Field(0.0, ge=0.0, le=1.0, description="Extractor confidence 0–1")
    extractor_used:   str           = Field("unknown")


# ─── Rule-based extractor ─────────────────────────────────────────────────────

EQUIPMENT_TAG_RE = re.compile(r'\b([A-Z]{1,4}[-]\d{2,4})\b')

FAILURE_MODE_KEYWORDS = {
    "bearing wear":             ["bearing", "bear ", "brg"],
    "seal leak":                ["seal leak", "seal fail", "mech seal", "mechanical seal"],
    "shaft misalignment":       ["misalign", "alignment"],
    "motor winding failure":    ["winding", "motor fail", "burnout", "rewound", "rewind"],
    "VFD fault":                ["vfd", "drive fault", "variable frequ"],
    "sensor fault":             ["sensor fault", "sensor fail", "transmitter fail"],
    "hydraulic pressure loss":  ["pressure loss", "low pressure", "pressure drop"],
    "valve malfunction":        ["valve fail", "valve stuck", "valve malfunction"],
    "transmitter drift":        ["transmitter drift", "4-20", "signal drift"],
    "thermocouple failure":     ["thermocouple", "thermowell", "rtd fail"],
    "scheduled PM":             ["scheduled pm", "annual pm", "preventive maint", "pm perform"],
    "operator damage":          ["forklift", "operator error", "improper oper", "impact damage"],
}

PART_PATTERNS = [
    re.compile(r'replaced?\s+([a-z][\w\s\-]{2,40}?)(?:\s+and|\s+on|\s*[,.]|$)', re.I),
    re.compile(r'parts\s+used[:\s]+([a-z][\w\s\-,]{2,60}?)(?:\.|$)', re.I),
    re.compile(r'consumables\s+used[:\s]+([a-z][\w\s\-,]{2,60}?)(?:\.|$)', re.I),
    re.compile(r'installed\s+(?:new\s+)?([a-z][\w\s\-]{2,40}?)(?:\s+and|\s+on|\s*[,.]|$)', re.I),
]

ROOT_CAUSE_PATTERNS = [
    re.compile(r'root\s+cause[:\s]+([^.]{5,120})\.?', re.I),
    re.compile(r'caused?\s+by[:\s]+([^.]{5,100})\.?', re.I),
    re.compile(r'due\s+to[:\s]+([^.]{5,100})\.?', re.I),
    re.compile(r'cause[:\s]+([^.]{5,100})\.?', re.I),
]

CATEGORY_KEYWORDS = {
    "mechanical_failure":      ["bearing", "seal", "misalign", "vibrat", "impeller", "shaft", "coupling", "rotor"],
    "electrical_failure":      ["motor", "winding", "vfd", "overload", "contactor", "wiring", "electrical", "amps", "phase"],
    "hydraulic_failure":       ["hydraulic", "hose", "cylinder", "accumulator", "fluid contamination", "pressure relief"],
    "instrumentation_failure": ["transmitter", "thermocouple", "sensor", "calibration", "4-20", "rtd", "impulse line", "flow meter"],
    "preventive_maintenance":  ["scheduled pm", "annual pm", "preventive maint", "lube route", "runtime hour"],
    "operator_damage":         ["forklift", "operator error", "improper", "impact damage", "over-tighten"],
}


def _rule_extract(text: str) -> WorkOrderFields:
    t = text.lower()

    tag_match = EQUIPMENT_TAG_RE.search(text)
    eq_tag = tag_match.group(1) if tag_match else None

    failure_mode = None
    for mode, keywords in FAILURE_MODE_KEYWORDS.items():
        if any(kw in t for kw in keywords):
            failure_mode = mode
            break

    parts = None
    for pat in PART_PATTERNS:
        m = pat.search(text)
        if m:
            candidate = m.group(1).strip().lower()
            if 3 < len(candidate) < 60:
                parts = candidate
                break

    root_cause = None
    for pat in ROOT_CAUSE_PATTERNS:
        m = pat.search(text)
        if m:
            root_cause = m.group(1).strip()
            break

    category = None
    cat_scores: dict[str, int] = {}
    for cat, keywords in CATEGORY_KEYWORDS.items():
        cat_scores[cat] = sum(1 for kw in keywords if kw in t)
    if cat_scores:
        best_cat = max(cat_scores, key=lambda c: cat_scores[c])
        if cat_scores[best_cat] > 0:
            category = best_cat

    confidence = sum([
        0.3 if eq_tag else 0,
        0.2 if failure_mode else 0,
        0.2 if parts else 0,
        0.15 if root_cause else 0,
        0.15 if category else 0,
    ])

    return WorkOrderFields(
        equipment_tag=eq_tag,
        failure_mode=failure_mode,
        parts_replaced=parts,
        root_cause=root_cause,
        failure_category=category,
        confidence=round(confidence, 2),
        extractor_used="rule_based",
    )


# ─── LLM extractor ────────────────────────────────────────────────────────────

_FEW_SHOT = """
Work order: "Responded to high bearing temperature alarm on P-104 (centrifugal pump). Investigation found bearing wear. Root cause: inadequate lubrication. Replaced mechanical seal and bearing set. Equipment returned to service."
{
  "equipment_tag": "P-104",
  "failure_mode": "bearing wear",
  "parts_replaced": "mechanical seal and bearing set",
  "root_cause": "inadequate lubrication",
  "failure_category": "mechanical_failure"
}

Work order: "WO raised for M-210. Motor tripped on overcurrent — found phase C reading 0 amps. Burning smell from motor. Replaced burned motor windings — sent to motor shop for rewind. Verified operation after reinstall."
{
  "equipment_tag": "M-210",
  "failure_mode": "motor winding failure",
  "parts_replaced": "motor winding (rewound)",
  "root_cause": "overheating due to blocked cooling fins",
  "failure_category": "electrical_failure"
}

Work order: "Completed annual PM on P-302. No anomalies noted during inspection. Replaced coupling insert and V-belt set as scheduled. All measurements within specification."
{
  "equipment_tag": "P-302",
  "failure_mode": "scheduled PM — no failure",
  "parts_replaced": "coupling insert and V-belt set",
  "root_cause": "calendar-based preventive maintenance interval reached",
  "failure_category": "preventive_maintenance"
}
""".strip()


def _llm_extract(text: str, retry: int = 2) -> WorkOrderFields:
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package required for LLM extraction: pip install openai")

    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    schema = {
        "type": "object",
        "properties": {
            "equipment_tag":    {"type": ["string", "null"]},
            "failure_mode":     {"type": ["string", "null"]},
            "parts_replaced":   {"type": ["string", "null"]},
            "root_cause":       {"type": ["string", "null"]},
            "failure_category": {
                "type": ["string", "null"],
                "enum": ["mechanical_failure", "electrical_failure", "hydraulic_failure",
                         "instrumentation_failure", "preventive_maintenance", "operator_damage", None],
            },
        },
        "required": ["equipment_tag", "failure_mode", "parts_replaced", "root_cause", "failure_category"],
        "additionalProperties": False,
    }

    prompt = (
        f"Extract structured fields from this maintenance work order. "
        f"Return JSON matching the schema exactly.\n\n"
        f"Examples:\n{_FEW_SHOT}\n\n"
        f"Work order: \"{text}\""
    )

    for attempt in range(retry + 1):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_schema", "json_schema": {"name": "work_order_fields", "schema": schema, "strict": True}},
                temperature=0,
                max_tokens=256,
            )
            import json
            data = json.loads(resp.choices[0].message.content)
            return WorkOrderFields(**data, confidence=0.95, extractor_used="llm")
        except Exception:
            if attempt == retry:
                raise
            time.sleep(1.0 * (attempt + 1))


# ─── ETLExtractor (main interface) ────────────────────────────────────────────

class ETLExtractor:
    """
    Pluggable extraction backend for maintenance work orders.

    Args:
        mode: "rule_based" | "llm" | "hybrid"
        llm_confidence_threshold: in hybrid mode, records below this rule-based
            confidence score are escalated to LLM (default 0.7)
    """

    def __init__(self, mode: str = "hybrid", llm_confidence_threshold: float = 0.70):
        if mode not in ("rule_based", "llm", "hybrid"):
            raise ValueError("mode must be 'rule_based', 'llm', or 'hybrid'")
        self.mode = mode
        self.llm_confidence_threshold = llm_confidence_threshold

    def extract(self, text: str) -> WorkOrderFields:
        if self.mode == "rule_based":
            return _rule_extract(text)
        if self.mode == "llm":
            return _llm_extract(text)
        # hybrid
        rule_result = _rule_extract(text)
        if rule_result.confidence < self.llm_confidence_threshold:
            try:
                llm_result = _llm_extract(text)
                llm_result.extractor_used = "hybrid_llm"
                return llm_result
            except Exception:
                rule_result.extractor_used = "hybrid_rule_fallback"
                return rule_result
        rule_result.extractor_used = "hybrid_rule"
        return rule_result

    def batch_extract(
        self,
        texts: list[str],
        delay_ms: float = 100,
    ) -> list[WorkOrderFields]:
        """Extract from a list of texts. Applies rate-limiting delay for LLM calls."""
        results = []
        for i, text in enumerate(texts):
            results.append(self.extract(text))
            if self.mode in ("llm", "hybrid") and i % 10 == 9:
                time.sleep(delay_ms / 1000)
        return results
