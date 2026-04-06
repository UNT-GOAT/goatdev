"""
Grade Calculator - Converts measurements + weight to grade

Uses three inputs per animal:
  1. Live weight (lbs)
  2. Condition Index (CI) = avg_body_width_cm / withers_height_cm
     Measures overall body fill — wider relative to height = better condition
  3. Muscle Distribution Ratio (MDR) = rump_width_cm / waist_width_cm
     Measures hindquarter muscle development — higher = more muscular rear

Grade tiers are defined per category (species + description). An animal must
meet ALL THREE minimums (weight, CI, MDR) to qualify for a tier. Tiers are
checked top-down from best to worst. If no tier is met, the animal receives
"No Roll" (below market floor).

These thresholds are preliminary — calibrated from facility operator knowledge
and initial measurements. They will be refined as more graded animals with
validated manual grades are collected.
"""

from typing import Dict, Optional, List, Tuple
from .logger import log
from .config import VALID_GRADES


# ============================================================
# GRADE TIER TABLES
#
# Each category maps to a list of (grade, min_weight, min_ci, min_mdr)
# tuples ordered from BEST to WORST. First match wins.
#
# CI  = avg_body_width_cm / withers_height_cm   (Width/Height)
# MDR = rump_width_cm / waist_width_cm          (Rump/Waist)
# ============================================================

GRADE_TIERS: Dict[str, List[Tuple[str, float, float, float]]] = {
    # Goat: Meat
    'goat:meat': [
        ('Reserve',    105, 0.60, 1.06),
        ('CAB Prime',  105, 0.60, 1.06),
        ('Prime',       90, 0.55, 1.02),
        ('CAB Choice',  82, 0.52, 1.00),
        ('Choice',      75, 0.48, 0.98),
        ('Select',      45, 0.40, 0.90),
    ],
    # Goat: Cross
    'goat:cross': [
        ('CAB Prime',  100, 0.58, 1.04),
        ('Prime',       85, 0.53, 1.00),
        ('Choice',      70, 0.46, 0.96),
        ('Select',      45, 0.40, 0.90),
    ],
    # Goat: Dairy
    'goat:dairy': [
        ('Prime',      110, 0.50, 1.00),
        ('Choice',      85, 0.45, 0.96),
        ('Select',      50, 0.40, 0.92),
    ],
    # Lamb: Lamb
    'lamb:lamb': [
        ('Reserve',    125, 0.62, 1.10),
        ('CAB Prime',  125, 0.62, 1.10),
        ('Prime',      110, 0.58, 1.05),
        ('Choice',      95, 0.52, 1.00),
        ('Select',      80, 0.45, 0.95),
    ],
    # Lamb: Ewe
    'lamb:ewe': [
        ('Prime',      150, 0.55, 1.02),
        ('Choice',     130, 0.50, 0.98),
        ('Select',     100, 0.44, 0.94),
    ],
}


def calculate_grade(
    measurements: Dict,
    live_weight_lbs: float,
    serial_id: str,
    species: str = None,
    description: str = None,
) -> Dict:
    """
    Calculate grade from measurements and weight.

    Args:
        measurements: Dict with withers_height_cm, avg_body_width_cm,
                     rump_width_cm, waist_width_cm (from grader)
        live_weight_lbs: Live weight in pounds
        serial_id: For logging
        species: 'goat' or 'lamb'
        description: 'meat', 'dairy', 'cross', 'lamb', 'ewe'

    Returns:
        Dict with:
            grade: str — the assigned grade
            details: dict — reasoning with computed ratios, category,
                     tier comparison, and deficits
    """
    log.info('grade', 'Calculating grade',
             serial_id=serial_id, weight=live_weight_lbs,
             species=species, description=description)

    details = {
        'weight_lbs': live_weight_lbs,
        'species': species,
        'description': description,
    }

    # --- Compute CI (Condition Index) ---
    withers_height = measurements.get('withers_height_cm')
    avg_width = measurements.get('avg_body_width_cm')

    ci = None
    if withers_height and withers_height > 0 and avg_width and avg_width > 0:
        ci = round(avg_width / withers_height, 3)
    else:
        log.warn('grade', 'Cannot compute CI — missing withers_height or avg_body_width',
                 serial_id=serial_id,
                 withers_height=withers_height,
                 avg_body_width=avg_width)

    details['ci'] = ci
    details['ci_inputs'] = {
        'avg_body_width_cm': avg_width,
        'withers_height_cm': withers_height,
    }

    # --- Compute MDR (Muscle Distribution Ratio) ---
    rump_width = measurements.get('rump_width_cm')
    waist_width = measurements.get('waist_width_cm')

    mdr = None
    if rump_width and rump_width > 0 and waist_width and waist_width > 0:
        mdr = round(rump_width / waist_width, 3)
    else:
        log.warn('grade', 'Cannot compute MDR — missing rump_width or waist_width',
                 serial_id=serial_id,
                 rump_width=rump_width,
                 waist_width=waist_width)

    details['mdr'] = mdr
    details['mdr_inputs'] = {
        'rump_width_cm': rump_width,
        'waist_width_cm': waist_width,
    }

    log.info('grade', 'Computed ratios',
             serial_id=serial_id, ci=ci, mdr=mdr, weight=live_weight_lbs)

    # --- Look up tier table ---
    category = _resolve_category(species, description)
    details['category'] = category

    tiers = GRADE_TIERS.get(category)

    if not tiers:
        log.warn('grade', 'No tier table for category, falling back to weight-only',
                 serial_id=serial_id, category=category)
        grade = _fallback_weight_only(live_weight_lbs, serial_id)
        details['method'] = 'weight_only_fallback'
        details['reason'] = f'No tier table for category {category}'
        return {'grade': grade, 'details': details}

    # Build tier comparison table so the operator can see where they land
    tier_comparison = []
    for tier_grade, min_weight, min_ci, min_mdr in tiers:
        entry = {
            'tier': tier_grade,
            'min_weight': min_weight,
            'min_ci': min_ci,
            'min_mdr': min_mdr,
            'weight_ok': live_weight_lbs >= min_weight,
        }
        if ci is not None:
            entry['ci_ok'] = bool(ci >= min_ci)
            entry['ci_delta'] = round(float(ci - min_ci), 3)
        else:
            entry['ci_ok'] = None
            entry['ci_delta'] = None

        if mdr is not None:
            entry['mdr_ok'] = bool(mdr >= min_mdr)
            entry['mdr_delta'] = round(float(mdr - min_mdr), 3)
        else:
            entry['mdr_ok'] = None
            entry['mdr_delta'] = None

        tier_comparison.append(entry)

    details['tier_comparison'] = tier_comparison

    # --- Full algorithm (all three inputs available) ---
    if ci is not None and mdr is not None:
        details['method'] = 'full'
        for tier_grade, min_weight, min_ci, min_mdr in tiers:
            if (live_weight_lbs >= min_weight and
                    ci >= min_ci and
                    mdr >= min_mdr):
                log.info('grade', 'Grade determined (full algorithm)',
                         serial_id=serial_id, grade=tier_grade,
                         category=category, ci=ci, mdr=mdr,
                         weight=live_weight_lbs)
                details['reason'] = f'Meets all minimums for {tier_grade}'
                return {'grade': tier_grade, 'details': details}

        # Below all tiers — find what's closest (lowest tier = Select)
        lowest = tiers[-1]
        deficits = {}
        if live_weight_lbs < lowest[1]:
            deficits['weight'] = round(lowest[1] - live_weight_lbs, 1)
        if ci < lowest[2]:
            deficits['ci'] = round(lowest[2] - ci, 3)
        if mdr < lowest[3]:
            deficits['mdr'] = round(lowest[3] - mdr, 3)

        details['deficits_from_select'] = deficits
        details['reason'] = f'Below Select floor — deficits: {deficits}'

        log.info('grade', 'Below all tiers — No Roll',
                 serial_id=serial_id, category=category,
                 ci=ci, mdr=mdr, weight=live_weight_lbs,
                 deficits=deficits)
        return {'grade': 'No Roll', 'details': details}

    # --- Partial data: use weight + whichever ratio we have ---
    details['method'] = 'partial'
    log.warn('grade', 'Partial measurements — grading with available data',
             serial_id=serial_id, has_ci=ci is not None, has_mdr=mdr is not None)

    missing = []
    if ci is None:
        missing.append('CI (missing avg_body_width or withers_height)')
    if mdr is None:
        missing.append('MDR (missing rump_width or waist_width)')
    details['missing_ratios'] = missing

    for tier_grade, min_weight, min_ci, min_mdr in tiers:
        weight_ok = live_weight_lbs >= min_weight
        ci_ok = ci >= min_ci if ci is not None else True
        mdr_ok = mdr >= min_mdr if mdr is not None else True

        if weight_ok and ci_ok and mdr_ok:
            log.info('grade', 'Grade determined (partial — missing ratios skipped)',
                     serial_id=serial_id, grade=tier_grade,
                     category=category, ci=ci, mdr=mdr,
                     weight=live_weight_lbs)
            details['reason'] = (
                f'Meets available minimums for {tier_grade} '
                f'(missing ratios treated as passing)'
            )
            return {'grade': tier_grade, 'details': details}

    details['reason'] = 'Below all tiers even with missing ratios skipped'
    return {'grade': 'No Roll', 'details': details}


def _resolve_category(species: str, description: str) -> str:
    """
    Map species + description to a tier table key.

    Handles missing/unknown values gracefully with sensible defaults.
    """
    species = (species or '').lower().strip()
    description = (description or '').lower().strip()

    if species == 'goat':
        if description in ('meat', 'cross', 'dairy'):
            return f'goat:{description}'
        return 'goat:meat'

    if species == 'lamb':
        if description in ('lamb', 'ewe'):
            return f'lamb:{description}'
        return 'lamb:lamb'

    # Unknown species — default to goat:meat
    return 'goat:meat'


def _fallback_weight_only(live_weight_lbs: float, serial_id: str) -> str:
    """
    Last-resort fallback when no tier table matches.
    Simple weight-based grade — same as the old placeholder.
    """
    if live_weight_lbs >= 105:
        grade = 'Prime'
    elif live_weight_lbs >= 80:
        grade = 'Choice'
    elif live_weight_lbs >= 60:
        grade = 'Select'
    else:
        grade = 'No Roll'

    log.info('grade', 'Grade from weight-only fallback',
             serial_id=serial_id, grade=grade, weight=live_weight_lbs)
    return grade


def validate_grade(grade: str) -> bool:
    """Check if a grade value is valid."""
    return grade in VALID_GRADES