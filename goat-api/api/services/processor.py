"""
Processing Service

Handles the actual YOLO model inference and measurement extraction.
Adapts your existing model code to work as a service.
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

MODEL_BASE = Path(os.getenv("MODEL_DIR", "/app/model"))

sys.path.insert(0, str(MODEL_BASE))
sys.path.insert(0, str(MODEL_BASE / "side"))
sys.path.insert(0, str(MODEL_BASE / "top"))
sys.path.insert(0, str(MODEL_BASE / "front"))

from .s3 import S3Service


class ProcessingService:
    def __init__(self, s3_service: S3Service, db_service):
        self.s3 = s3_service
        self.db = db_service
        
        # Lazy load models (they're heavy)
        self._side_model = None
        self._top_model = None
        self._front_model = None
        self._models_loaded = False
    
    def _load_models(self):
        """Load YOLO models and calibrations (lazy loading)"""
        if self._models_loaded:
            return
        
        print("Loading YOLO models...")
        
        # Import model classes
        from side_yolo_measurements import YOLOGoatMeasurementsSide # type: ignore
        from top_yolo_measurements import YOLOGoatMeasurementsTop # type: ignore
        from front_yolo_measurements import YOLOGoatMeasurementsFront # type: ignore
        
        # Load calibrations
        with open(MODEL_BASE / "side" / "side_calibration.json", "r") as f:
            side_cal = json.load(f)
        
        with open(MODEL_BASE / "top" / "top_calibration.json", "r") as f:
            top_cal = json.load(f)
        
        with open(MODEL_BASE / "front" / "front_calibration.json", "r") as f:
            front_cal = json.load(f)
        
        # Initialize models
        self._side_model = YOLOGoatMeasurementsSide(
            str(MODEL_BASE / "side" / "best.pt"),
            side_cal["pixels_per_cm"]
        )
        self._side_model.debug = True
        
        self._top_model = YOLOGoatMeasurementsTop(
            str(MODEL_BASE / "top" / "best.pt"),
            top_cal["pixels_per_cm"]
        )
        self._top_model.debug = True
        
        self._front_model = YOLOGoatMeasurementsFront(
            str(MODEL_BASE / "front" / "best.pt"),
            front_cal["pixels_per_cm"]
        )
        self._front_model.debug = True
        
        self._models_loaded = True
        print("✓ Models loaded")
    
    async def process_goat_images(self, timestamp: str, goat_id: int) -> Dict[str, Any]:
        """
        Main processing function.
        
        1. Download images from S3
        2. Run through YOLO models
        3. Extract measurements
        4. Upload annotated images
        5. Save to database
        6. Return results
        """
        
        # Ensure models are loaded
        self._load_models()
        
        # Create temp directory for processing
        work_dir = tempfile.mkdtemp(prefix=f"goat_{timestamp}_")
        debug_dir = os.path.join(work_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        
        try:
            # Download images from S3
            print(f"Downloading images for {timestamp}...")
            images = self.s3.download_images(timestamp, work_dir)
            
            if len(images) < 3:
                raise ValueError(f"Expected 3 images, got {len(images)}: {list(images.keys())}")
            
            print(f"Downloaded: {list(images.keys())}")
            
            # Process each view
            results = {
                "timestamp": timestamp,
                "goat_id": goat_id,
                "processed_at": datetime.now().isoformat(),
                "measurements": {},
                "confidence_scores": {},
                "view_results": {},
                "errors": []
            }
            
            annotated_paths = {}
            
            # Side view
            if "side" in images:
                print("Processing side view...")
                side_result = self._side_model.process_image(images["side"])
                results["view_results"]["side"] = side_result
                
                if side_result.get("success"):
                    results["measurements"]["head_height_cm"] = side_result.get("head_height_cm")
                    results["measurements"]["withers_height_cm"] = side_result.get("withers_height_cm")
                    results["measurements"]["rump_height_cm"] = side_result.get("rump_height_cm")
                    results["confidence_scores"]["side"] = side_result.get("yolo_confidence")
                    
                    # Get debug image path
                    if "debug_image" in side_result:
                        annotated_paths["side"] = side_result["debug_image"]
                else:
                    results["errors"].append(f"Side: {side_result.get('error', 'Unknown error')}")
            
            # Top view
            if "top" in images:
                print("Processing top view...")
                top_result = self._top_model.process_image(images["top"])
                results["view_results"]["top"] = top_result
                
                if top_result.get("success"):
                    results["measurements"]["top_body_width_cm"] = top_result.get("body_width_cm")
                    results["confidence_scores"]["top"] = top_result.get("yolo_confidence")
                    
                    if "debug_image" in top_result:
                        annotated_paths["top"] = top_result["debug_image"]
                else:
                    results["errors"].append(f"Top: {top_result.get('error', 'Unknown error')}")
            
            # Front view
            if "front" in images:
                print("Processing front view...")
                front_result = self._front_model.process_image(images["front"])
                results["view_results"]["front"] = front_result
                
                if front_result.get("success"):
                    results["measurements"]["front_body_width_cm"] = front_result.get("body_width_cm")
                    results["confidence_scores"]["front"] = front_result.get("yolo_confidence")
                    
                    if "debug_image" in front_result:
                        annotated_paths["front"] = front_result["debug_image"]
                else:
                    results["errors"].append(f"Front: {front_result.get('error', 'Unknown error')}")
            
            # Calculate average body width
            widths = [
                results["measurements"].get("top_body_width_cm"),
                results["measurements"].get("front_body_width_cm")
            ]
            valid_widths = [w for w in widths if w is not None]
            if valid_widths:
                results["measurements"]["avg_body_width_cm"] = round(
                    sum(valid_widths) / len(valid_widths), 2
                )
            
            # Determine overall success
            results["all_views_successful"] = len(results["errors"]) == 0
            results["success"] = len(results["confidence_scores"]) > 0
            
            # Upload annotated images to S3
            print("Uploading annotated images...")
            self.s3.upload_annotated_images(timestamp, annotated_paths)
            
            # Upload results JSON
            self.s3.upload_results_json(timestamp, results)
            
            # Save to database
            print("Saving to database...")
            self.db.save_measurement(goat_id, timestamp, results["measurements"])
            
            # Clean up view_results (too verbose for response)
            del results["view_results"]
            
            print(f"✓ Processing complete for {timestamp}")
            return results
            
        finally:
            # Clean up temp directory
            shutil.rmtree(work_dir, ignore_errors=True)
            
            # Clean up model debug directories
            for view in ["side", "top", "front"]:
                debug_path = MODEL_BASE / view / "debug"
                if debug_path.exists():
                    shutil.rmtree(debug_path, ignore_errors=True)
