"""
Grade Calculator - Converts measurements + weight to grade

This is a placeholder implementation. The actual grading logic
needs to be determined based on the goat grading standards.

Grades (from best to worst):
- Reserve
- CAB Prime
- Prime
- CAB Choice
- Choice
- Select
- No Roll
"""

from typing import Dict, Optional
from .logger import log
from .config import VALID_GRADES


def calculate_grade(
    measurements: Dict,
    live_weight_lbs: float,
    serial_id: str
) -> Optional[str]:
    """
    Calculate goat grade from measurements and weight.
    
    Args:
        measurements: Dict with head_height_cm, withers_height_cm, rump_height_cm,
                     top_body_width_cm, front_body_width_cm, avg_body_width_cm
        live_weight_lbs: Live weight in pounds
        serial_id: For logging
        
    Returns:
        Grade string or None if cannot calculate
    """
    log.info('grade', 'Calculating grade', serial_id=serial_id, weight=live_weight_lbs)
    
    # Check we have minimum required measurements
    required = ['withers_height_cm', 'avg_body_width_cm']
    missing = [m for m in required if not measurements.get(m)]
    
    if missing:
        log.warn('grade', 'Missing measurements for grade calculation',
                serial_id=serial_id, missing=','.join(missing))
        return None
    
    # TODO: Implement actual grading logic
    # For now, return a placeholder based on weight ranges
    # This is NOT the real grading algorithm
    
    # Placeholder logic (REPLACE WITH REAL ALGORITHM)
    if live_weight_lbs >= 80:
        grade = "Choice"
    elif live_weight_lbs >= 60:
        grade = "Select"
    else:
        grade = "No Roll"
    
    log.info('grade', 'Grade calculated (placeholder)',
            serial_id=serial_id, grade=grade)
    
    return grade


def validate_grade(grade: str) -> bool:
    """Check if a grade value is valid"""
    return grade in VALID_GRADES
