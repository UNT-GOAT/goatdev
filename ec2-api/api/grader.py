"""
Goat Grader - Wraps YOLO models for measurement extraction

Handles model loading, inference, and measurement extraction for all three views.

Measurement strategy:
- Side view: height measurements (head, withers, rump) from mask thirds
- Top view: width measurements (shoulder, waist, rump) using anatomical proportions
- Front view: chest width from mask bounding box
"""

import cv2
import numpy as np
import json
import time
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field

from .logger import log
from .config import (
    SIDE_MODEL_PATH, TOP_MODEL_PATH, FRONT_MODEL_PATH,
    SIDE_CALIBRATION_PATH, TOP_CALIBRATION_PATH, FRONT_CALIBRATION_PATH,
    MIN_CONFIDENCE_THRESHOLD, WARN_CONFIDENCE_THRESHOLD,
    MIN_MEASUREMENT_CM, MAX_MEASUREMENT_CM,
    MODEL_INFERENCE_TIMEOUT_SEC,
    TOP_RUMP_PCT, TOP_SHOULDER_PCT,
    TOP_VIEW_DIRECTION
)


@dataclass
class ViewResult:
    """Result from processing a single view"""
    success: bool
    measurements: Dict
    confidence: Optional[float] = None
    error: Optional[str] = None
    fix: Optional[str] = None
    warnings: List[str] = None
    mask: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class GoatGrader:
    """
    Manages all three YOLO models and performs goat measurement extraction.

    Thread-safe: Uses locks for model inference.

    Processing:
    - Side view: heights from mask divided into thirds
    - Top view: widths at anatomical proportions (configurable percentages from tail end)
    - Front view: chest width from bounding box
    """

    def __init__(self):
        self.side_model = None
        self.top_model = None
        self.front_model = None

        self.side_pixels_per_cm = None
        self.top_pixels_per_cm = None
        self.front_pixels_per_cm = None

        self._inference_lock = threading.Lock()
        self._initialized = False
        self._gpu_available = False

    def initialize(self) -> Tuple[bool, List[str]]:
        """
        Load all models and calibrations.

        Returns:
            (success, list of errors)
        """
        errors = []

        log.info('grader:init', 'Starting model initialization')

        # Check for GPU
        try:
            import torch
            self._gpu_available = torch.cuda.is_available()
            if self._gpu_available:
                gpu_name = torch.cuda.get_device_name(0)
                log.info('grader:init', 'GPU available', device=gpu_name)
            else:
                log.warn('grader:init', 'No GPU available, using CPU',
                        fix='Install CUDA for faster inference')
        except Exception as e:
            log.warn('grader:init', 'Could not check GPU', error=str(e))
            self._gpu_available = False

        # Load calibrations first
        calibration_errors = self._load_calibrations()
        errors.extend(calibration_errors)

        if calibration_errors:
            log.error('grader:init', 'Calibration loading failed',
                     errors=len(calibration_errors))
            return False, errors

        # Load models
        model_errors = self._load_models()
        errors.extend(model_errors)

        if model_errors:
            log.error('grader:init', 'Model loading failed',
                     errors=len(model_errors))
            return False, errors

        self._initialized = True
        log.info('grader:init', 'Initialization complete',
                gpu=self._gpu_available)

        return True, []

    def _load_calibrations(self) -> List[str]:
        """Load calibration files for all views"""
        errors = []

        # Side calibration
        try:
            if not SIDE_CALIBRATION_PATH.exists():
                errors.append(f"Side calibration not found: {SIDE_CALIBRATION_PATH}")
                log.error('grader:calibration', 'Side calibration file missing',
                         path=str(SIDE_CALIBRATION_PATH),
                         fix='Ensure model/side/side_calibration.json exists')
            else:
                with open(SIDE_CALIBRATION_PATH, 'r') as f:
                    cal = json.load(f)
                    self.side_pixels_per_cm = cal.get('pixels_per_cm')
                    if not self.side_pixels_per_cm or self.side_pixels_per_cm <= 0:
                        errors.append("Invalid side calibration: pixels_per_cm must be positive")
                    else:
                        log.info('grader:calibration', 'Side calibration loaded',
                                pixels_per_cm=round(self.side_pixels_per_cm, 2))
        except json.JSONDecodeError as e:
            errors.append(f"Side calibration JSON invalid: {e}")
            log.error('grader:calibration', 'Side calibration parse error',
                     error=str(e), fix='Check JSON syntax in side_calibration.json')
        except Exception as e:
            errors.append(f"Side calibration error: {e}")
            log.exception('grader:calibration', 'Side calibration failed', error=str(e))

        # Top calibration
        try:
            if not TOP_CALIBRATION_PATH.exists():
                errors.append(f"Top calibration not found: {TOP_CALIBRATION_PATH}")
                log.error('grader:calibration', 'Top calibration file missing',
                         path=str(TOP_CALIBRATION_PATH),
                         fix='Ensure model/top/top_calibration.json exists')
            else:
                with open(TOP_CALIBRATION_PATH, 'r') as f:
                    cal = json.load(f)
                    self.top_pixels_per_cm = cal.get('pixels_per_cm')
                    if not self.top_pixels_per_cm or self.top_pixels_per_cm <= 0:
                        errors.append("Invalid top calibration: pixels_per_cm must be positive")
                    else:
                        log.info('grader:calibration', 'Top calibration loaded',
                                pixels_per_cm=round(self.top_pixels_per_cm, 2))
        except json.JSONDecodeError as e:
            errors.append(f"Top calibration JSON invalid: {e}")
        except Exception as e:
            errors.append(f"Top calibration error: {e}")

        # Front calibration
        try:
            if not FRONT_CALIBRATION_PATH.exists():
                errors.append(f"Front calibration not found: {FRONT_CALIBRATION_PATH}")
                log.error('grader:calibration', 'Front calibration file missing',
                         path=str(FRONT_CALIBRATION_PATH),
                         fix='Ensure model/front/front_calibration.json exists')
            else:
                with open(FRONT_CALIBRATION_PATH, 'r') as f:
                    cal = json.load(f)
                    self.front_pixels_per_cm = cal.get('pixels_per_cm')
                    if not self.front_pixels_per_cm or self.front_pixels_per_cm <= 0:
                        errors.append("Invalid front calibration: pixels_per_cm must be positive")
                    else:
                        log.info('grader:calibration', 'Front calibration loaded',
                                pixels_per_cm=round(self.front_pixels_per_cm, 2))
        except json.JSONDecodeError as e:
            errors.append(f"Front calibration JSON invalid: {e}")
        except Exception as e:
            errors.append(f"Front calibration error: {e}")

        return errors

    def _load_models(self) -> List[str]:
        """Load YOLO models for all views"""
        errors = []

        try:
            from ultralytics import YOLO
        except ImportError as e:
            errors.append(f"ultralytics not installed: {e}")
            log.error('grader:model', 'ultralytics package missing',
                     error=str(e), fix='pip install ultralytics')
            return errors

        # Side model
        try:
            if not SIDE_MODEL_PATH.exists():
                errors.append(f"Side model not found: {SIDE_MODEL_PATH}")
                log.error('grader:model', 'Side model file missing',
                         path=str(SIDE_MODEL_PATH),
                         fix='Ensure model/side/best.pt exists')
            else:
                log.info('grader:model:side', 'Loading side model')
                self.side_model = YOLO(str(SIDE_MODEL_PATH))
                log.info('grader:model:side', 'Side model loaded')
        except Exception as e:
            errors.append(f"Side model failed to load: {e}")
            log.exception('grader:model:side', 'Side model load failed', error=str(e))

        # Top model
        try:
            if not TOP_MODEL_PATH.exists():
                errors.append(f"Top model not found: {TOP_MODEL_PATH}")
                log.error('grader:model', 'Top model file missing',
                         path=str(TOP_MODEL_PATH),
                         fix='Ensure model/top/best.pt exists')
            else:
                log.info('grader:model:top', 'Loading top model')
                self.top_model = YOLO(str(TOP_MODEL_PATH))
                log.info('grader:model:top', 'Top model loaded')
        except Exception as e:
            errors.append(f"Top model failed to load: {e}")
            log.exception('grader:model:top', 'Top model load failed', error=str(e))

        # Front model
        try:
            if not FRONT_MODEL_PATH.exists():
                errors.append(f"Front model not found: {FRONT_MODEL_PATH}")
                log.error('grader:model', 'Front model file missing',
                         path=str(FRONT_MODEL_PATH),
                         fix='Ensure model/front/best.pt exists')
            else:
                log.info('grader:model:front', 'Loading front model')
                self.front_model = YOLO(str(FRONT_MODEL_PATH))
                log.info('grader:model:front', 'Front model loaded')
        except Exception as e:
            errors.append(f"Front model failed to load: {e}")
            log.exception('grader:model:front', 'Front model load failed', error=str(e))

        return errors

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def gpu_available(self) -> bool:
        return self._gpu_available

    def process_images(
        self,
        side_image: np.ndarray,
        top_image: np.ndarray,
        front_image: np.ndarray,
        serial_id: str,
        generate_debug_images: bool = True
    ) -> Dict:
        """
        Process all three images and return combined measurements.

        Each view is independent — no cross-view dependencies.

        Args:
            side_image: BGR image from side camera
            top_image: BGR image from top camera
            front_image: BGR image from front camera
            serial_id: Goat identifier for logging
            generate_debug_images: If True, generate annotated debug images

        Returns:
            Dictionary with measurements, confidence scores, errors, warnings, debug_images
        """
        if not self._initialized:
            log.error('grader:process', 'Grader not initialized', serial_id=serial_id)
            return {
                'success': False,
                'error': 'Grader not initialized',
                'fix': 'Restart the API server'
            }

        start_time = time.time()
        log.info('grader:process', 'Starting analysis', serial_id=serial_id)

        results = {
            'measurements': {},
            'confidence_scores': {},
            'view_errors': [],
            'warnings': [],
            'all_views_successful': True,
            'success': True,
            'debug_images': None
        }

        side_result = None
        top_result = None
        front_result = None

        with self._inference_lock:
            # 1. Side view - heights
            side_result = self._process_side(side_image, serial_id)
            if side_result.success:
                results['measurements'].update(side_result.measurements)
                results['confidence_scores']['side'] = side_result.confidence
                results['warnings'].extend(side_result.warnings)
            else:
                results['all_views_successful'] = False
                results['view_errors'].append({
                    'view': 'side',
                    'error': side_result.error,
                    'fix': side_result.fix
                })

            # 2. Top view - widths at anatomical positions
            top_result = self._process_top(top_image, serial_id)
            if top_result.success:
                results['measurements'].update(top_result.measurements)
                results['confidence_scores']['top'] = top_result.confidence
                results['warnings'].extend(top_result.warnings)
            else:
                results['all_views_successful'] = False
                results['view_errors'].append({
                    'view': 'top',
                    'error': top_result.error,
                    'fix': top_result.fix
                })

            # 3. Front view - chest width
            front_result = self._process_front(front_image, serial_id)
            if front_result.success:
                results['measurements'].update(front_result.measurements)
                results['confidence_scores']['front'] = front_result.confidence
                results['warnings'].extend(front_result.warnings)
            else:
                results['all_views_successful'] = False
                results['view_errors'].append({
                    'view': 'front',
                    'error': front_result.error,
                    'fix': front_result.fix
                })

        # Calculate average body width if we have measurements
        shoulder_width = results['measurements'].get('shoulder_width_cm')
        front_width = results['measurements'].get('front_body_width_cm')

        if shoulder_width and front_width:
            results['measurements']['avg_body_width_cm'] = round(
                (shoulder_width + front_width) / 2, 2
            )
        elif shoulder_width:
            results['measurements']['avg_body_width_cm'] = shoulder_width
            results['warnings'].append('avg_body_width based on top view only')
        elif front_width:
            results['measurements']['avg_body_width_cm'] = front_width
            results['warnings'].append('avg_body_width based on front view only')

        # Generate debug images
        if generate_debug_images:
            try:
                debug_images = self.draw_debug_overlay(
                    side_image, top_image, front_image,
                    side_result.mask if side_result else None,
                    top_result.mask if top_result else None,
                    front_result.mask if front_result else None,
                    results['measurements'],
                    serial_id
                )
                results['debug_images'] = debug_images
            except Exception as e:
                log.exception('grader:debug', 'Failed to generate debug images',
                             serial_id=serial_id, error=str(e))
                results['debug_images'] = None

        # Clean up empty lists
        if not results['view_errors']:
            results['view_errors'] = None
        if not results['warnings']:
            results['warnings'] = None

        duration = round(time.time() - start_time, 2)

        log.info('grader:process', 'Analysis complete',
                serial_id=serial_id,
                all_views_ok=results['all_views_successful'],
                duration_sec=duration)

        return results

    # =========================================================================
    # SIDE VIEW
    # =========================================================================

    def _process_side(self, image: np.ndarray, serial_id: str) -> ViewResult:
        """
        Process side view image.

        Extracts height measurements by dividing mask into thirds.
        """
        log.info('grader:model:side', 'Processing side view', serial_id=serial_id)

        try:
            results = self.side_model.predict(
                source=image,
                conf=MIN_CONFIDENCE_THRESHOLD,
                verbose=False
            )

            if len(results) == 0 or results[0].masks is None:
                log.warn('grader:model:side', 'No goat detected', serial_id=serial_id)
                return ViewResult(
                    success=False,
                    measurements={},
                    error='No goat detected in side image',
                    fix='Ensure goat is fully visible in side camera view'
                )

            detection = results[0]
            if len(detection.boxes) == 0:
                return ViewResult(
                    success=False,
                    measurements={},
                    error='No goat detected in side image',
                    fix='Ensure goat is fully visible in side camera view'
                )

            confidence = float(detection.boxes[0].conf)
            warnings = []

            if confidence < WARN_CONFIDENCE_THRESHOLD:
                warnings.append(f'Low confidence on side view: {confidence:.2f}')
                log.warn('grader:model:side', 'Low confidence detection',
                        serial_id=serial_id, confidence=confidence)

            # Get mask and resize to image dimensions
            mask = detection.masks[0].data[0].cpu().numpy()
            mask_resized = cv2.resize(
                mask,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
            mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255

            # Check mask is not too small
            mask_area = np.sum(mask_binary > 0)
            image_area = image.shape[0] * image.shape[1]
            mask_ratio = mask_area / image_area

            if mask_ratio < 0.01:
                log.warn('grader:model:side', 'Mask too small',
                        serial_id=serial_id, mask_ratio=round(mask_ratio, 4))
                return ViewResult(
                    success=False,
                    measurements={},
                    confidence=confidence,
                    error='Detected region too small',
                    fix='Move goat closer to camera or check camera alignment'
                )

            # Extract height measurements
            measurements = self._extract_side_measurements(mask_binary)

            if not measurements:
                return ViewResult(
                    success=False,
                    measurements={},
                    confidence=confidence,
                    error='Failed to extract measurements from mask',
                    fix='Image may be corrupted or goat not properly positioned'
                )

            # Sanity check measurements
            for key, value in measurements.items():
                if value and (value < MIN_MEASUREMENT_CM or value > MAX_MEASUREMENT_CM):
                    warnings.append(f'{key} outside expected range: {value}cm')
                    log.warn('grader:model:side', 'Measurement outside range',
                            serial_id=serial_id, measurement=key, value=value)

            log.info('grader:model:side', 'Side view complete',
                    serial_id=serial_id, confidence=round(confidence, 3))

            return ViewResult(
                success=True,
                measurements=measurements,
                confidence=confidence,
                warnings=warnings,
                mask=mask_binary
            )

        except Exception as e:
            log.exception('grader:model:side', 'Side processing failed',
                         serial_id=serial_id, error=str(e))
            return ViewResult(
                success=False,
                measurements={},
                error=f'Side processing error: {str(e)}',
                fix='Check logs for details, may be a model or memory issue'
            )

    def _extract_side_measurements(self, mask: np.ndarray) -> Dict:
        """Extract height measurements from side view mask"""
        measurements = {}

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return measurements

        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)

        # Ground level = bottom of bounding box
        ground_level = y + h

        # Divide into thirds: front (head), middle (withers), rear (rump)
        third = w // 3

        # Head height (front third)
        front_section = [point[0] for point in contour if x <= point[0][0] < x + third]
        if front_section and len(front_section) > 5:
            front_points = np.array(front_section)
            head_top = np.min(front_points[:, 1])
            head_height = ground_level - head_top
            measurements['head_height_cm'] = round(head_height / self.side_pixels_per_cm, 2)

        # Withers height (middle third)
        middle_section = [point[0] for point in contour if x + third <= point[0][0] < x + 2*third]
        if middle_section and len(middle_section) > 5:
            middle_points = np.array(middle_section)
            top_half_cutoff = y + (h * 0.30)
            top_half_points = middle_points[middle_points[:, 1] <= top_half_cutoff]

            if len(top_half_points) > 0:
                y_coords = top_half_points[:, 1]
                y_sorted = np.sort(y_coords)[::-1]
                num_points = min(5, len(y_sorted))
                withers_top = int(np.mean(y_sorted[:num_points]))
            else:
                withers_top = np.min(middle_points[:, 1])

            withers_height = ground_level - withers_top
            measurements['withers_height_cm'] = round(withers_height / self.side_pixels_per_cm, 2)

        # Rump height (rear third)
        rear_section = [point[0] for point in contour if x + 2*third <= point[0][0] <= x + w]
        if rear_section and len(rear_section) > 5:
            rear_points = np.array(rear_section)
            rump_top = np.min(rear_points[:, 1])
            rump_height = ground_level - rump_top
            measurements['rump_height_cm'] = round(rump_height / self.side_pixels_per_cm, 2)

        return measurements

    # =========================================================================
    # TOP VIEW
    # =========================================================================

    def _process_top(self, image: np.ndarray, serial_id: str) -> ViewResult:
        """
        Process top view image.

        Measures widths at anatomical positions (fixed percentages from tail end).
        """
        log.info('grader:model:top', 'Processing top view', serial_id=serial_id)

        try:
            results = self.top_model.predict(
                source=image,
                conf=MIN_CONFIDENCE_THRESHOLD,
                verbose=False
            )

            if len(results) == 0 or results[0].masks is None:
                log.warn('grader:model:top', 'No goat detected', serial_id=serial_id)
                return ViewResult(
                    success=False,
                    measurements={},
                    error='No goat detected in top image',
                    fix='Ensure goat is fully visible in top camera view'
                )

            detection = results[0]
            if len(detection.boxes) == 0:
                return ViewResult(
                    success=False,
                    measurements={},
                    error='No goat detected in top image',
                    fix='Ensure goat is fully visible in top camera view'
                )

            confidence = float(detection.boxes[0].conf)
            warnings = []

            if confidence < WARN_CONFIDENCE_THRESHOLD:
                warnings.append(f'Low confidence on top view: {confidence:.2f}')

            # Get body mask
            mask = detection.masks[0].data[0].cpu().numpy()
            mask_resized = cv2.resize(
                mask,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
            mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255

            # Extract width measurements
            measurements = self._extract_top_measurements(mask_binary, serial_id)

            if not measurements:
                return ViewResult(
                    success=False,
                    measurements={},
                    confidence=confidence,
                    error='Failed to extract measurements from top mask',
                    fix='Check camera angle and goat positioning'
                )

            log.info('grader:model:top', 'Top view complete',
                    serial_id=serial_id, confidence=round(confidence, 3))

            return ViewResult(
                success=True,
                measurements=measurements,
                confidence=confidence,
                warnings=warnings,
                mask=mask_binary
            )

        except Exception as e:
            log.exception('grader:model:top', 'Top processing failed',
                         serial_id=serial_id, error=str(e))
            return ViewResult(
                success=False,
                measurements={},
                error=f'Top processing error: {str(e)}',
                fix='Check logs for details'
            )

    def _extract_top_measurements(self, mask: np.ndarray, serial_id: str) -> Dict:
        """
        Extract width measurements from top view mask.

        Uses fixed anatomical proportions from the tail (left) end of the mask.
        Rump and shoulder positions are configured in config.py as percentages
        of the mask bounding box width. Waist is the midpoint.
        """
        measurements = {}

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return measurements

        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)

        def get_width_at_x(target_x: int) -> float:
            """Get the width of the mask at a specific x position (vertical slice)"""
            if target_x < 0 or target_x >= mask.shape[1]:
                return 0

            column = mask[:, target_x]
            y_coords = np.where(column > 0)[0]

            if len(y_coords) < 2:
                return 0

            width_pixels = np.max(y_coords) - np.min(y_coords)
            return width_pixels

        # Calculate positions from anatomical percentages
        # TOP_RUMP_PCT and TOP_SHOULDER_PCT are defined as distance from the TAIL end.
        # TOP_VIEW_DIRECTION tells us which side the head is on, so we know where
        # the tail end is (the opposite side).
        if TOP_VIEW_DIRECTION == 'right':
            # Head on right, tail on left — percentages measured from left edge
            rump_x = int(x + TOP_RUMP_PCT * w)
            shoulder_x = int(x + TOP_SHOULDER_PCT * w)
        else:
            # Head on left, tail on right — percentages measured from right edge
            rump_x = int(x + (1.0 - TOP_RUMP_PCT) * w)
            shoulder_x = int(x + (1.0 - TOP_SHOULDER_PCT) * w)
        waist_x = int((rump_x + shoulder_x) / 2)

        log.info('grader:top:widths', 'Anatomical position mapping',
                serial_id=serial_id,
                rump_pct=TOP_RUMP_PCT,
                shoulder_pct=TOP_SHOULDER_PCT,
                rump_x=rump_x,
                waist_x=waist_x,
                shoulder_x=shoulder_x,
                top_bbox_x=x,
                top_bbox_w=w)

        # Get widths at each position (average over small window for stability)
        window = 5  # pixels on each side

        # Shoulder width
        shoulder_widths = [get_width_at_x(shoulder_x + i) for i in range(-window, window+1)]
        shoulder_widths = [sw for sw in shoulder_widths if sw > 0]
        if shoulder_widths:
            shoulder_width = np.mean(shoulder_widths) / self.top_pixels_per_cm
            measurements['shoulder_width_cm'] = round(shoulder_width, 2)

        # Waist width
        waist_widths = [get_width_at_x(waist_x + i) for i in range(-window, window+1)]
        waist_widths = [ww for ww in waist_widths if ww > 0]
        if waist_widths:
            waist_width = np.mean(waist_widths) / self.top_pixels_per_cm
            measurements['waist_width_cm'] = round(waist_width, 2)

        # Rump width
        rump_widths = [get_width_at_x(rump_x + i) for i in range(-window, window+1)]
        rump_widths = [rw for rw in rump_widths if rw > 0]
        if rump_widths:
            rump_width = np.mean(rump_widths) / self.top_pixels_per_cm
            measurements['rump_width_cm'] = round(rump_width, 2)

        log.info('grader:top:widths', 'Width measurements extracted',
                serial_id=serial_id,
                shoulder=measurements.get('shoulder_width_cm'),
                waist=measurements.get('waist_width_cm'),
                rump=measurements.get('rump_width_cm'))

        return measurements

    # =========================================================================
    # FRONT VIEW
    # =========================================================================

    def _process_front(self, image: np.ndarray, serial_id: str) -> ViewResult:
        """Process front view image - chest/body width"""
        log.info('grader:model:front', 'Processing front view', serial_id=serial_id)

        try:
            results = self.front_model.predict(
                source=image,
                conf=MIN_CONFIDENCE_THRESHOLD,
                verbose=False
            )

            if len(results) == 0 or results[0].masks is None:
                log.warn('grader:model:front', 'No goat detected', serial_id=serial_id)
                return ViewResult(
                    success=False,
                    measurements={},
                    error='No goat detected in front image',
                    fix='Ensure goat is fully visible in front camera view'
                )

            detection = results[0]
            if len(detection.boxes) == 0:
                return ViewResult(
                    success=False,
                    measurements={},
                    error='No goat detected in front image',
                    fix='Ensure goat is fully visible in front camera view'
                )

            confidence = float(detection.boxes[0].conf)
            warnings = []

            if confidence < WARN_CONFIDENCE_THRESHOLD:
                warnings.append(f'Low confidence on front view: {confidence:.2f}')

            mask = detection.masks[0].data[0].cpu().numpy()
            mask_resized = cv2.resize(
                mask,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
            mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255

            measurements = self._extract_front_measurements(mask_binary)

            if not measurements:
                return ViewResult(
                    success=False,
                    measurements={},
                    confidence=confidence,
                    error='Failed to extract measurements from front mask',
                    fix='Check camera angle and goat positioning'
                )

            log.info('grader:model:front', 'Front view complete',
                    serial_id=serial_id, confidence=round(confidence, 3))

            return ViewResult(
                success=True,
                measurements=measurements,
                confidence=confidence,
                warnings=warnings,
                mask=mask_binary
            )

        except Exception as e:
            log.exception('grader:model:front', 'Front processing failed',
                         serial_id=serial_id, error=str(e))
            return ViewResult(
                success=False,
                measurements={},
                error=f'Front processing error: {str(e)}',
                fix='Check logs for details'
            )

    def _extract_front_measurements(self, mask: np.ndarray) -> Dict:
        """Extract width measurement from front view mask (chest width)"""
        measurements = {}

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return measurements

        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)

        body_width = w / self.front_pixels_per_cm
        measurements['front_body_width_cm'] = round(body_width, 2)

        return measurements

    # =========================================================================
    # DEBUG OVERLAY
    # =========================================================================

    def draw_debug_overlay(
        self,
        side_image: np.ndarray,
        top_image: np.ndarray,
        front_image: np.ndarray,
        side_mask: Optional[np.ndarray],
        top_mask: Optional[np.ndarray],
        front_mask: Optional[np.ndarray],
        measurements: Dict,
        serial_id: str
    ) -> Dict[str, np.ndarray]:
        """
        Draw debug visualization on images.

        Returns:
            Dictionary with 'side', 'top', 'front' debug images
        """
        debug_images = {}

        # Colors
        MASK_COLOR = (0, 255, 0)      # Green - mask outline
        SHOULDER_COLOR = (255, 0, 0)   # Blue - shoulder
        WAIST_COLOR = (0, 255, 255)    # Yellow - waist
        RUMP_COLOR = (0, 0, 255)       # Red - rump
        TEXT_BG = (0, 0, 0)            # Black background for text

        def draw_text_with_bg(img, text, pos, color, scale=0.6):
            """Draw text with black background for readability"""
            font = cv2.FONT_HERSHEY_SIMPLEX
            thickness = 2
            (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            tx, ty = pos
            cv2.rectangle(img, (tx - 2, ty - th - 4), (tx + tw + 2, ty + 4), TEXT_BG, -1)
            cv2.putText(img, text, (tx, ty), font, scale, color, thickness)

        # =================================================================
        # SIDE VIEW DEBUG
        # =================================================================
        if side_image is not None:
            side_debug = side_image.copy()

            if side_mask is not None:
                contours, _ = cv2.findContours(side_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cv2.drawContours(side_debug, contours, -1, MASK_COLOR, 2)

                    contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(contour)

                    # Draw bounding box
                    cv2.rectangle(side_debug, (x, y), (x + w, y + h), (128, 128, 128), 1)

            # Add title
            draw_text_with_bg(side_debug, f'SIDE VIEW - {serial_id}', (10, side_debug.shape[0] - 20), (255, 255, 255), 0.7)

            debug_images['side'] = side_debug

        # =================================================================
        # TOP VIEW DEBUG
        # =================================================================
        if top_image is not None:
            top_debug = top_image.copy()

            if top_mask is not None:
                contours, _ = cv2.findContours(top_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cv2.drawContours(top_debug, contours, -1, MASK_COLOR, 2)

                    contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(contour)

                    # Draw bounding box
                    cv2.rectangle(top_debug, (x, y), (x + w, y + h), (128, 128, 128), 1)

                    # Calculate measurement positions from anatomical percentages
                    if TOP_VIEW_DIRECTION == 'right':
                        # Head on right, tail on left
                        rump_x = int(x + TOP_RUMP_PCT * w)
                        shoulder_x = int(x + TOP_SHOULDER_PCT * w)
                    else:
                        # Head on left, tail on right
                        rump_x = int(x + (1.0 - TOP_RUMP_PCT) * w)
                        shoulder_x = int(x + (1.0 - TOP_SHOULDER_PCT) * w)
                    waist_x = int((rump_x + shoulder_x) / 2)

                    line_extend = 30

                    # Shoulder line and measurement
                    cv2.line(top_debug, (shoulder_x, y - line_extend), (shoulder_x, y + h + line_extend), SHOULDER_COLOR, 3)
                    if measurements.get('shoulder_width_cm'):
                        draw_text_with_bg(top_debug, f"Shoulder: {measurements['shoulder_width_cm']}cm",
                                         (shoulder_x - 60, y - line_extend - 10), SHOULDER_COLOR)

                    # Waist line and measurement
                    cv2.line(top_debug, (waist_x, y - line_extend), (waist_x, y + h + line_extend), WAIST_COLOR, 3)
                    if measurements.get('waist_width_cm'):
                        draw_text_with_bg(top_debug, f"Waist: {measurements['waist_width_cm']}cm",
                                         (waist_x - 50, y - line_extend - 10), WAIST_COLOR)

                    # Rump line and measurement
                    cv2.line(top_debug, (rump_x, y - line_extend), (rump_x, y + h + line_extend), RUMP_COLOR, 3)
                    if measurements.get('rump_width_cm'):
                        draw_text_with_bg(top_debug, f"Rump: {measurements['rump_width_cm']}cm",
                                         (rump_x - 40, y - line_extend - 10), RUMP_COLOR)

            # Add title
            draw_text_with_bg(top_debug, f'TOP VIEW - {serial_id}', (10, top_debug.shape[0] - 20), (255, 255, 255), 0.7)

            debug_images['top'] = top_debug

        # =================================================================
        # FRONT VIEW DEBUG
        # =================================================================
        if front_image is not None:
            front_debug = front_image.copy()

            if front_mask is not None:
                contours, _ = cv2.findContours(front_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cv2.drawContours(front_debug, contours, -1, MASK_COLOR, 2)

                    contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(contour)

                    # Draw bounding box
                    cv2.rectangle(front_debug, (x, y), (x + w, y + h), (128, 128, 128), 1)

                    # Draw width measurement line
                    mid_y = y + h // 2
                    cv2.line(front_debug, (x, mid_y), (x + w, mid_y), SHOULDER_COLOR, 3)
                    cv2.circle(front_debug, (x, mid_y), 8, SHOULDER_COLOR, -1)
                    cv2.circle(front_debug, (x + w, mid_y), 8, SHOULDER_COLOR, -1)

                    if measurements.get('front_body_width_cm'):
                        draw_text_with_bg(front_debug, f"Chest Width: {measurements['front_body_width_cm']}cm",
                                         (x, mid_y - 20), SHOULDER_COLOR)

            # Add title
            draw_text_with_bg(front_debug, f'FRONT VIEW - {serial_id}', (10, front_debug.shape[0] - 20), (255, 255, 255), 0.7)

            debug_images['front'] = front_debug

        log.info('grader:debug', 'Debug images generated', serial_id=serial_id, count=len(debug_images))

        return debug_images

    def run_test_inference(self) -> Tuple[bool, float]:
        """Run a quick test inference to verify models are working."""
        if not self._initialized:
            return False, 0.0

        try:
            test_image = np.zeros((100, 100, 3), dtype=np.uint8)

            start = time.time()

            with self._inference_lock:
                self.side_model.predict(source=test_image, conf=0.1, verbose=False)

            duration_ms = (time.time() - start) * 1000

            return True, round(duration_ms, 2)

        except Exception as e:
            log.error('grader:test', 'Test inference failed', error=str(e))
            return False, 0.0


# Global grader instance
grader = GoatGrader()