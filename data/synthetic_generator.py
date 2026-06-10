"""
data/synthetic_generator.py — Synthetic maintenance work order corpus generator.

Generates 3,000 realistic maintenance work orders across 6 failure categories.
Each record has two parallel artifacts:
  (a) Raw narrative text — what a technician would write in a CMMS
  (b) Ground-truth structured fields — for ETL evaluation in notebook 02

Domain authority: Vocabulary and failure taxonomy drawn from 12+ years of industrial
engineering across HVAC (Rheem), subsea (Centurion), and manufacturing (Daikin/Baker Hughes).

Usage:
    python data/synthetic_generator.py
    # Saves: data/work_orders.csv (raw text for NLP)
    #        data/work_orders_ground_truth.json (structured fields for ETL eval)

Output schema:
    work_order_id, date, equipment_tag, equipment_type, technician_id,
    failure_category, text, labor_hours, parts_used,
    [ground_truth: failure_mode, parts_replaced, root_cause]
"""

from __future__ import annotations
import random
import json
import uuid
from datetime import date, timedelta
from pathlib import Path
import pandas as pd

# ─── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)

N_RECORDS = 3000
OUT_DIR = Path("data")

# ─── Equipment taxonomy ───────────────────────────────────────────────────────
EQUIPMENT_TYPES = {
    "pump":         ["P", ["centrifugal pump", "process pump", "booster pump", "chemical injection pump"]],
    "compressor":   ["C", ["reciprocating compressor", "screw compressor", "air compressor", "gas compressor"]],
    "motor":        ["M", ["induction motor", "drive motor", "fan motor", "pump motor"]],
    "valve":        ["V", ["control valve", "gate valve", "ball valve", "check valve", "pressure relief valve"]],
    "heat_exchanger":["HX", ["shell-and-tube heat exchanger", "plate heat exchanger", "fin-fan cooler"]],
    "instrument":   ["TT", ["thermocouple", "pressure transmitter", "flow meter", "level transmitter", "RTD sensor"]],
    "conveyor":     ["CV", ["belt conveyor", "screw conveyor", "chain conveyor"]],
    "fan":          ["FN", ["induced draft fan", "forced draft fan", "cooling tower fan", "exhaust fan"]],
    "agitator":     ["AG", ["tank agitator", "mixer", "blender"]],
    "boiler":       ["BL", ["fire-tube boiler", "water-tube boiler", "steam generator"]],
}

TECHNICIAN_IDS = [f"TECH-{i:03d}" for i in range(1, 31)]

AREAS = ["Unit A", "Unit B", "Compressor Station 1", "Pump House 2",
         "Utilities Area", "Production Floor", "Maintenance Bay", "Substation 4"]

# ─── Category vocabulary and templates ────────────────────────────────────────

FAILURE_CATEGORIES = [
    "mechanical_failure",
    "electrical_failure",
    "hydraulic_failure",
    "instrumentation_failure",
    "preventive_maintenance",
    "operator_damage",
]

# Per-category structured content pools
_MECH = {
    "failure_modes": [
        "bearing wear", "seal leak", "shaft misalignment", "coupling failure",
        "impeller wear", "rotor imbalance", "vibration excessive", "shaft seal failure",
        "mechanical seal degradation", "wear ring failure", "bushing wear",
    ],
    "root_causes": [
        "normal wear due to extended run time", "inadequate lubrication",
        "contaminated lubricant", "cavitation damage", "misalignment during last PM",
        "corrosion from process fluid", "fatigue crack propagation",
        "operating beyond design parameters", "installation error",
    ],
    "parts_replaced": [
        "mechanical seal", "bearing set", "coupling", "impeller",
        "wear ring", "shaft sleeve", "o-ring set", "gasket set",
        "lip seal", "bushing", "bearing housing",
    ],
    "actions": [
        "Replaced mechanical seal and bearing set",
        "Realigned shaft coupling and replaced worn bearings",
        "Rebuilt pump — replaced impeller, wear rings, and mechanical seal",
        "Replaced shaft seal assembly and flushed out contaminants",
        "Balanced rotor and replaced coupling insert",
    ],
    "observations": [
        "Noted excessive vibration during routine rounds",
        "Operator reported unusual noise from equipment",
        "Process leak detected at seal face — fluid on floor",
        "High bearing temperature alarm triggered in DCS",
        "Vibration monitor exceeded 5 mm/s alarm setpoint",
    ],
}

_ELEC = {
    "failure_modes": [
        "motor winding failure", "insulation breakdown", "sensor fault",
        "VFD fault", "overload trip", "phase imbalance", "terminal connection failure",
        "contactor failure", "PLC I/O fault", "wiring damage",
    ],
    "root_causes": [
        "moisture ingress into motor terminal box", "overheating due to blocked cooling fins",
        "voltage surge from upstream fault", "vibration-induced terminal loosening",
        "insulation aging — motor at end of service life", "rodent damage to wiring",
        "incorrect VFD parameters after recent upgrade",
    ],
    "parts_replaced": [
        "motor winding (rewound)", "motor terminal block", "VFD drive unit",
        "overload relay", "contactor assembly", "temperature sensor", "proximity switch",
        "cable termination lug", "control fuse", "PLC I/O module",
    ],
    "actions": [
        "Replaced burned motor windings — sent to motor shop for rewind",
        "Replaced VFD drive unit and re-programmed with correct parameters",
        "Replaced failed contactor and tightened all terminal connections",
        "Repaired wiring damage and re-ran conduit section",
        "Replaced motor temperature sensor and verified alarm setpoints",
    ],
    "observations": [
        "Motor tripped on overcurrent — found phase C reading 0 amps",
        "Burning smell from motor — IR scan showed hot spot on winding",
        "VFD fault code F-028 — overcurrent during acceleration",
        "Operator reported equipment not starting from DCS",
        "SCADA alarm: high motor temperature 145°C vs. 120°C setpoint",
    ],
}

_HYD = {
    "failure_modes": [
        "hydraulic pressure loss", "valve malfunction", "cylinder seal failure",
        "fluid contamination", "pump cavitation", "line rupture", "accumulator failure",
        "relief valve chattering",
    ],
    "root_causes": [
        "deteriorated hose assembly — fatigue crack at fitting",
        "contaminated fluid — particle count ISO 18/16/13",
        "cylinder seal hardened from high operating temperature",
        "incorrect fluid viscosity after topping up with wrong grade",
        "relief valve set point drifted above design",
        "pump inlet strainer plugged",
    ],
    "parts_replaced": [
        "hydraulic hose assembly", "cylinder seal kit", "directional control valve",
        "hydraulic pump", "accumulator bladder", "pressure relief valve",
        "inlet strainer", "hydraulic filter element", "check valve",
    ],
    "actions": [
        "Replaced ruptured hydraulic hose and flushed system",
        "Replaced cylinder seal kit — cleaned rod and inspected bore",
        "Replaced directional control valve — cleaned manifold block",
        "Drained and replaced hydraulic fluid — flushed system with clean oil",
        "Replaced plugged inlet strainer and checked pump inlet pressure",
    ],
    "observations": [
        "Slow cylinder response — measured system pressure 1200 psi vs. 2500 psi design",
        "Visible hydraulic fluid leak at cylinder rod seal",
        "Actuator hunting on pressure control — relief valve chattering",
        "Fluid sample shows NAS 12 contamination — above NAS 9 limit",
        "Pump making cavitation noise — inlet vacuum at 12 inHg",
    ],
}

_INST = {
    "failure_modes": [
        "transmitter drift", "thermocouple failure", "flow meter fouling",
        "sensor signal loss", "calibration out of tolerance", "impulse line plugging",
        "level transmitter false reading", "RTD open circuit",
    ],
    "root_causes": [
        "impulse line plugged with process buildup",
        "thermowell corroded — thermocouple insulation compromised",
        "flow meter orifice plate corroded and undersized",
        "moisture in junction box causing signal noise",
        "process condensate in impulse line causing offset error",
        "EMI interference from nearby VFD installation",
    ],
    "parts_replaced": [
        "pressure transmitter", "thermocouple and thermowell",
        "orifice plate", "RTD element", "level transmitter",
        "junction box terminal strip", "instrument isolation valve",
        "transmitter manifold",
    ],
    "actions": [
        "Replaced pressure transmitter and flushed impulse lines",
        "Replaced thermocouple and thermowell — calibrated in bench loop",
        "Replaced orifice plate and recalculated flow coefficient",
        "Calibrated level transmitter against tape measure reference",
        "Sealed junction box and replaced corroded terminals",
    ],
    "observations": [
        "DCS reading 245 psi — gauge reads 185 psi — 60 psi deviation",
        "Temperature reading frozen at 78°C — no response to process change",
        "Flow meter reading 0 gpm during normal operation",
        "Noisy 4–20 mA signal — spikes visible on historian trend",
        "Level transmitter reading 85% with vessel visually at 40%",
    ],
}

_PM = {
    "failure_modes": ["scheduled PM — no failure", "wear items at life limit"],
    "root_causes": [
        "calendar-based preventive maintenance interval reached",
        "runtime-based PM trigger — 8760 hours", "annual shutdown scope",
    ],
    "parts_replaced": [
        "lubrication oil and filter", "coupling insert", "belt drive set",
        "mechanical seal (preventive)", "bearing set (preventive)",
        "air filter", "spark plugs", "V-belts", "coolant",
    ],
    "actions": [
        "Completed annual PM — changed oil, inspected all rotating parts",
        "Replaced coupling insert and V-belt set as scheduled",
        "Lubricated all grease points per lube route card",
        "Performed 1000-hour inspection — no defects found",
        "Replaced mechanical seal as part of scheduled overhaul",
    ],
    "observations": [
        "Equipment operating normally — PM performed per schedule",
        "No anomalies noted during inspection",
        "All measurements within specification",
    ],
}

_OPDMG = {
    "failure_modes": [
        "impact damage from forklift", "improper operation", "overload — exceeded design rating",
        "over-tightened flange — cracked body", "dropped equipment",
        "opened valve against locked-out system", "wrong fluid loaded",
    ],
    "root_causes": [
        "forklift operator struck equipment during material handling",
        "operator bypassed interlock — operated outside safe limits",
        "incorrect procedure followed during startup",
        "excessive force applied during maintenance — thread stripped",
        "equipment operated with isolation valve closed",
    ],
    "parts_replaced": [
        "cracked valve body (replaced valve)", "bent shaft (straightened/replaced)",
        "damaged impeller", "broken coupling", "cracked flange",
    ],
    "actions": [
        "Replaced valve body damaged by forklift impact",
        "Straightened bent shaft — replaced if beyond tolerance",
        "Replaced broken coupling from overload event",
        "Repaired thread damage — replaced fitting",
        "Completed incident investigation and retraining",
    ],
    "observations": [
        "Physical damage to equipment casing — not process related",
        "Operator reported loud bang — found broken coupling",
        "Bent shaft discovered after vibration alarm — root cause: forklift impact",
        "Over-pressured vessel — PRV lifted — investigation initiated",
    ],
}

CATEGORY_DATA = {
    "mechanical_failure":       _MECH,
    "electrical_failure":       _ELEC,
    "hydraulic_failure":        _HYD,
    "instrumentation_failure":  _INST,
    "preventive_maintenance":   _PM,
    "operator_damage":          _OPDMG,
}

# ─── Text templates ───────────────────────────────────────────────────────────

def _generate_wo_text(eq_tag: str, eq_type: str, cat: str, cd: dict) -> tuple[str, dict]:
    """Generate one work order text and its ground-truth fields."""
    obs        = random.choice(cd["observations"])
    action     = random.choice(cd["actions"])
    part       = random.choice(cd["parts_replaced"])
    root_cause = random.choice(cd["root_causes"])
    failure_mode = random.choice(cd["failure_modes"])

    # Build narrative text (variable length, realistic technician style)
    templates = [
        (f"Responded to {obs.lower()} on {eq_tag} ({eq_type}). Investigation found {failure_mode}. "
         f"Root cause: {root_cause}. {action}. Parts used: {part}. "
         f"Equipment returned to service and tested satisfactory."),

        (f"WO raised for {eq_tag}. {obs} Found evidence of {failure_mode} on inspection. "
         f"{root_cause.capitalize()}. Corrective action: {action.lower()}. "
         f"Replaced {part}. Verified operation — no further issues."),

        (f"Maintenance performed on {eq_type} tag {eq_tag}. "
         f"Complaint: {obs.lower().replace('noted ', '').replace('operator reported ', '')}. "
         f"Diagnosis confirmed {failure_mode}. Cause determined to be {root_cause}. "
         f"Work performed: {action.lower()}. Consumables used: {part}. "
         f"Unit tested at design conditions post-repair."),
    ]

    # Add occasional abbreviations for realism
    abbrevs = {
        " mechanical seal ": " mech seal ", " bearing set ": " bearing kit ",
        " pressure transmitter ": " PT ", " temperature ": " temp ",
        " equipment ": " equip ", " maintenance ": " maint ",
    }
    text = random.choice(templates)
    if random.random() < 0.4:
        for full, abbr in abbrevs.items():
            if full in text:
                text = text.replace(full, abbr, 1)

    ground_truth = {
        "equipment_tag":  eq_tag,
        "failure_mode":   failure_mode,
        "parts_replaced": part,
        "root_cause":     root_cause,
        "failure_category": cat,
    }
    components = {"obs": obs, "action": action, "part": part,
                  "root_cause": root_cause, "failure_mode": failure_mode}
    text = _apply_noise(text, cat, cd, eq_tag, components, ground_truth)
    return text, ground_truth


# ─── Realism / noise layer ────────────────────────────────────────────────────
# Real CMMS text is messy: typos, shorthand, vague one-liners, overlapping
# symptoms, and occasional miscategorized records. Without this layer the
# per-category vocabulary is fully separable and every classifier scores 100%.

# Noise knobs (probabilities per record)
P_AMBIGUOUS_OBS  = 0.55   # replace category-specific observation with a generic one
P_SYMPTOM_BLEED  = 0.30   # borrow a failure mode mention from a confusable category
P_DROP_MODE      = 0.35   # technician never names the failure mode explicitly
P_GENERIC_ACTION = 0.30   # action described generically ("completed repair")
P_DROP_PART      = 0.25   # parts not itemized in text
P_TERSE          = 0.18   # collapse record to a terse one-liner
P_TYPOS          = 0.45   # introduce character-level typos
P_LOWERCASE      = 0.20   # technician wrote everything lowercase
P_FILLER         = 0.20   # append admin filler sentence
P_LABEL_NOISE    = 0.02   # miscategorized in CMMS (classification label only)

# Category-neutral action phrases — carry no classification signal
GENERIC_ACTIONS = [
    "completed repair as per procedure",
    "repaired and tested unit",
    "carried out corrective work and verified operation",
    "fault rectified — unit back online",
    "completed work order scope",
]

# Generic observations a technician could log for ANY failure type
AMBIGUOUS_OBSERVATIONS = [
    "Operator reported unusual noise from equipment",
    "Equipment tripped during normal operation",
    "Abnormal reading noted during routine rounds",
    "Unit shut down on alarm — cause not immediately obvious",
    "Leak observed near unit — source unclear at first inspection",
    "Performance degradation reported by operations",
    "Intermittent fault — could not reproduce on first visit",
    "Equipment found in failed state at shift start",
]

# Categories whose symptoms plausibly overlap in real plants
CONFUSION_NEIGHBORS = {
    "mechanical_failure":      ["hydraulic_failure", "operator_damage"],
    "electrical_failure":      ["instrumentation_failure"],
    "hydraulic_failure":       ["mechanical_failure"],
    "instrumentation_failure": ["electrical_failure"],
    "preventive_maintenance":  ["mechanical_failure", "instrumentation_failure"],
    "operator_damage":         ["mechanical_failure"],
}

FILLER_SENTENCES = [
    "See attached photos for reference.",
    "Follow-up WO raised for permanent repair.",
    "Parts ordered under PO 4500123 — temporary fix in place.",
    "Discussed with shift supervisor before closing WO.",
    "Awaiting engineering review before next PM cycle.",
    "No spares in stock — used refurbished unit from warehouse.",
]

SHORTHAND = {
    " mechanical seal": " mech seal", " bearing set": " brg set",
    " pressure transmitter": " PT", " temperature": " temp",
    " equipment": " equip", " maintenance": " maint",
    " replaced": " repl", " inspection": " inspx",
    " vibration": " vib", " hydraulic": " hyd",
    " investigation": " investig", " returned to service": " RTS",
}


def _typo(text: str, rate: float = 0.015) -> str:
    """Inject character-level typos (swap, drop, double) at ~rate per char."""
    chars = list(text)
    i = 0
    out = []
    while i < len(chars):
        c = chars[i]
        if c.isalpha() and random.random() < rate:
            op = random.random()
            if op < 0.4 and i + 1 < len(chars):      # swap with next
                out.append(chars[i + 1]); out.append(c); i += 2; continue
            elif op < 0.7:                            # drop
                i += 1; continue
            else:                                     # double
                out.append(c); out.append(c); i += 1; continue
        out.append(c); i += 1
    return "".join(out)


def _replace_any_case(text: str, target: str, repl: str) -> tuple[str, bool]:
    for variant in (target, target.lower(), target.capitalize()):
        if variant in text:
            return text.replace(variant, repl, 1), True
    return text, False


def _apply_noise(text: str, cat: str, cd: dict, eq_tag: str,
                 comp: dict, gt: dict) -> str:
    """Apply realism transformations. Mutates gt: when information is removed
    from the text, the corresponding ground-truth field is set to None so the
    ETL evaluation never penalizes an extractor for unextractable fields."""

    # 1. Terse one-liner: technician in a hurry. Most fields never make it
    # into the text.
    if random.random() < P_TERSE:
        part_short = comp["part"].split("(")[0].strip()
        action = (random.choice(GENERIC_ACTIONS) if random.random() < 0.5
                  else comp["action"].lower())
        keep_part = random.random() < 0.5
        if keep_part:
            text = f"{action} on {eq_tag}, repl {part_short}, tested ok"
        else:
            text = random.choice([
                f"{eq_tag} — {action}. ok now",
                f"{eq_tag} down. {action}. RTS",
            ])
            gt["parts_replaced"] = None
        gt["failure_mode"] = None
        gt["root_cause"] = None
    else:
        # 2. Replace category-specific observation with a generic one
        if random.random() < P_AMBIGUOUS_OBS:
            for variant in (comp["obs"], comp["obs"].lower()):
                if variant in text:
                    text = text.replace(
                        variant, random.choice(AMBIGUOUS_OBSERVATIONS).lower(), 1)
                    break

        # 3. Drop the explicit failure-mode mention
        if random.random() < P_DROP_MODE:
            text, found = _replace_any_case(text, comp["failure_mode"], "an issue")
            if found:
                gt["failure_mode"] = None

        # 4. Generic action wording — no category signal
        if random.random() < P_GENERIC_ACTION:
            text, _ = _replace_any_case(
                text, comp["action"], random.choice(GENERIC_ACTIONS))

        # 5. Parts not itemized
        if random.random() < P_DROP_PART:
            text, found = _replace_any_case(text, comp["part"], "misc consumables")
            if found:
                gt["parts_replaced"] = None

        # 6. Symptom bleed: mention a confusable category's failure mode
        if random.random() < P_SYMPTOM_BLEED:
            neighbor = random.choice(CONFUSION_NEIGHBORS[cat])
            bleed_mode = random.choice(CATEGORY_DATA[neighbor]["failure_modes"])
            text += f" Initially suspected {bleed_mode} but ruled out."

    # 7. Shorthand
    if random.random() < 0.7:
        for full, abbr in SHORTHAND.items():
            if full in text and random.random() < 0.5:
                text = text.replace(full, abbr)

    # 8. Admin filler
    if random.random() < P_FILLER:
        text += " " + random.choice(FILLER_SENTENCES)

    # 9. Typos and casing
    if random.random() < P_TYPOS:
        text = _typo(text)
    if random.random() < P_LOWERCASE:
        text = text.lower()

    return text


def generate_corpus(n: int = N_RECORDS, seed: int = SEED) -> tuple[pd.DataFrame, list[dict]]:
    """Generate n work orders. Returns (DataFrame of raw records, list of ground-truth dicts)."""
    random.seed(seed)
    records = []
    ground_truths = []

    # Balance classes roughly equally with small random noise
    base_per_cat = n // len(FAILURE_CATEGORIES)
    counts = {c: base_per_cat for c in FAILURE_CATEGORIES}
    remainder = n - base_per_cat * len(FAILURE_CATEGORIES)
    for c in list(FAILURE_CATEGORIES)[:remainder]:
        counts[c] += 1

    start_date = date(2019, 1, 1)
    end_date   = date(2024, 12, 31)
    date_range = (end_date - start_date).days

    eq_tags: dict[str, list[str]] = {}
    for etype, (prefix, _) in EQUIPMENT_TYPES.items():
        eq_tags[etype] = [f"{prefix}-{100 + i:03d}" for i in range(20)]

    wo_counter = 10000
    for cat in FAILURE_CATEGORIES:
        cat_data = CATEGORY_DATA[cat]
        for _ in range(counts[cat]):
            etype     = random.choice(list(EQUIPMENT_TYPES.keys()))
            eq_tag    = random.choice(eq_tags[etype])
            eq_name   = random.choice(EQUIPMENT_TYPES[etype][1])
            tech_id   = random.choice(TECHNICIAN_IDS)
            area      = random.choice(AREAS)
            wo_date   = start_date + timedelta(days=random.randint(0, date_range))
            labor_hrs = round(random.uniform(0.5, 8.0), 1)

            text, gt = _generate_wo_text(eq_tag, eq_name, cat, cat_data)

            # CMMS label noise: ~2% of records are miscategorized by the
            # technician. Ground truth (what actually happened) stays correct;
            # only the classification label column is flipped.
            label = cat
            if random.random() < P_LABEL_NOISE:
                label = random.choice(CONFUSION_NEIGHBORS[cat])

            part_for_inventory = gt["parts_replaced"] or random.choice(cat_data["parts_replaced"])
            part_short = part_for_inventory.split("(")[0].strip()
            qty = random.randint(1, 3)

            records.append({
                "work_order_id":   f"WO-{wo_counter}",
                "date":            wo_date.isoformat(),
                "area":            area,
                "equipment_tag":   eq_tag,
                "equipment_type":  eq_name,
                "technician_id":   tech_id,
                "failure_category": label,
                "text":            text,
                "labor_hours":     labor_hrs,
                "parts_used":      f"{qty}× {part_short}",
            })
            ground_truths.append({"work_order_id": f"WO-{wo_counter}", **gt})
            wo_counter += 1

    df = pd.DataFrame(records).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df, ground_truths


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print(f"[generator] Generating {N_RECORDS} work orders...")
    df, gts = generate_corpus()
    wo_csv = OUT_DIR / "work_orders.csv"
    gt_json = OUT_DIR / "work_orders_ground_truth.json"
    df.to_csv(wo_csv, index=False)
    gt_json.write_text(json.dumps(gts, indent=2))
    print(f"[generator] Saved {len(df):,} work orders → {wo_csv}")
    print(f"[generator] Saved ground truth → {gt_json}")
    print("\nClass distribution:")
    print(df.failure_category.value_counts().to_string())


if __name__ == "__main__":
    main()
