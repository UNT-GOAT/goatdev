
"""
Hi! Please read :P - Ethan

NOTE - HOW I GOT THIS WORKING:
I manually traced goat outlines on 23 images using a RoboFlow to create segmentation masks, then split them into 18 for training and 5 for validation. 
They are the usuable side images that we got from Becky on google photos.These are in /train and /val. 
Each of these has an /images and /labels subfolder. Images are the of images, and labes are the data containing the segmentation masks in YOLO format.
I then trained YOLOv8-seg for 10 epochs using these folder, which learned to recognize and segment goats from the labeled examples by fine-tuning pretrained weights. 
Now when you run this using a new image, YOLO will output a segmentation mask of the goat in the image using its weights from theses dirs.

NOTE - MEASUREMENT CALIBRATION: 
Currently using calibration.json for cm-to-pixel values. Temporarily, this json is set by calibration_tool.py, this takes
one of the side-goat pictures as input, allows the user to click on the marked cm distances on the glass, and takes the average pixels/cm value.
Until we have fully stationary and standardized image capture, this is necessary for decently accurate measurements.
This really only needs to be run once until we get more data, but still good to have around.

NOTE - OUTPUTS:
Outputs a side_yolo_measurements.json with measurements for each image processed in the followng format. A few of these fields will not be 
needed in our end product but they are good to have temporarily for guidance and debugging.

    side_yolo_measurements.json
        {
            "filename": "IMG_20251113_093457026_HDR.jpg",
            "success": true,
            "image_width": 4080,
            "image_height": 3072,
            "calibration_method": "manual",
            "pixels_per_cm": 24.34,
            "yolo_confidence": 0.327,
            "body_length_cm": 84.85,
            "length_to_height_ratio": 1.284,
            "head_height_cm": 66.07,
            "withers_height_cm": 55.55,
            "rump_height_cm": 60.15,
            "body_area_square_cm": 2491.45,
            "debug_image": "debug/debug_IMG_20251113_093457026_HDR.jpg"
        }

Also outputs a debug folder with images showing the segmentation mask and measurement lines drawn on the image for visual verification.
Check these out after each test run. Pretty neat.

Currently, with such a small data, we cannot expect high confidence scores. 
They do a fine job for now, but we can expect dramatic improvements with more training data.

"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional
from ultralytics import YOLO

class YOLOGoatMeasurements:
    def __init__(self, model_path: str, pixels_per_cm: Optional[float] = None):
        """
        Initialize YOLO measurement extractor
        
        Args:
            model_path: Path to trained YOLO segmentation model (.pt file)
            pixels_per_cm: Manual calibration value (from calibration_tool.py)
        """
        self.model = YOLO(model_path)
        self.manual_calibration = pixels_per_cm
        self.debug = False
    
    def extract_measurements(self, mask: np.ndarray, pixels_per_cm: float) -> Dict:
        """
        Extract body measurements from segmentation mask
        
        Args:
            mask: Binary mask of goat (from YOLO)
            pixels_per_cm: Calibration value
            
        Returns:
            Dictionary of essential measurements for grading
        """
        measurements = {}
        
        # Find contour from mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return measurements
        
        # Use largest contour
        contour = max(contours, key=cv2.contourArea)
        
        # Bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Body length (horizontal extent)
        measurements['body_length_cm'] = round(w / pixels_per_cm, 2)
        measurements['length_to_height_ratio'] = round(w / h, 3) if h > 0 else 0
        
        # Ground level = bottom of bounding box (lowest point of legs/hooves)
        ground_level = y + h
        
        # THREE HEIGHT MEASUREMENTS - from top of body to ground
        # Divide body into thirds horizontally: front (head), middle (withers), rear (rump)
        third = w // 3
        
        # Store measurement points for debug visualization
        measurements['_debug_points'] = {}
        
        # Front section (head area) - leftmost third
        front_section = [point[0] for point in contour if x <= point[0][0] < x + third]
        if front_section and len(front_section) > 5:
            front_points = np.array(front_section)
            # Find topmost point (head top)
            head_top = np.min(front_points[:, 1])
            # Measure from top to ground
            head_height = ground_level - head_top
            measurements['head_height_cm'] = round(head_height / pixels_per_cm, 2)
            # Store for debug
            measurements['_debug_points']['head'] = {
                'top': int(head_top),
                'bottom': int(ground_level),
                'x': int(x + third // 2)  # Middle of section
            }
        else:
            measurements['head_height_cm'] = 0.0
        
        # Middle section (withers/back area) - middle third
        middle_section = [point[0] for point in contour if x + third <= point[0][0] < x + 2*third]
        if middle_section and len(middle_section) > 5:
            middle_points = np.array(middle_section)
            # Define TOP 30% geometrically using bounding box
            top_half_cutoff = y + (h * 0.30)
            # Only consider points in the top half
            top_half_points = middle_points[middle_points[:, 1] <= top_half_cutoff]
            
            if len(top_half_points) > 0:
                # Get Y coordinates and sort to find lowest points
                y_coords = top_half_points[:, 1]
                y_sorted = np.sort(y_coords)[::-1]  # Sort descending (highest Y = lowest on screen)
                
                # Average the 5 lowest points (or fewer if not enough points)
                num_points = min(5, len(y_sorted))
                withers_top = int(np.mean(y_sorted[:num_points]))
            else:
                # Fallback to topmost point
                withers_top = np.min(middle_points[:, 1])
            
            # Measure from averaged lowest back points to ground
            withers_height = ground_level - withers_top
            measurements['withers_height_cm'] = round(withers_height / pixels_per_cm, 2)
            # Store for debug
            measurements['_debug_points']['withers'] = {
                'top': int(withers_top),
                'bottom': int(ground_level),
                'x': int(x + third + third // 2)
            }
        else:
            measurements['withers_height_cm'] = 0.0
        
        # Rear section (rump area) - rightmost third
        rear_section = [point[0] for point in contour if x + 2*third <= point[0][0] <= x + w]
        if rear_section and len(rear_section) > 5:
            rear_points = np.array(rear_section)
            # Find topmost point (rump top)
            rump_top = np.min(rear_points[:, 1])
            # Measure from top to ground
            rump_height = ground_level - rump_top
            measurements['rump_height_cm'] = round(rump_height / pixels_per_cm, 2)
            # Store for debug
            measurements['_debug_points']['rump'] = {
                'top': int(rump_top),
                'bottom': int(ground_level),
                'x': int(x + 2*third + third // 2)
            }
        else:
            measurements['rump_height_cm'] = 0.0
        
        # Body area
        area_pixels = cv2.contourArea(contour)
        area_square_cm = area_pixels / (pixels_per_cm ** 2)
        measurements['body_area_square_cm'] = round(area_square_cm, 2)
        
        return measurements
    
    def process_image(self, image_path: str, conf_threshold: float = 0.1) -> Dict:
        """
        Process a single image and extract measurements
        
        Args:
            image_path: Path to goat image
            conf_threshold: YOLO confidence threshold (0.1 works well)
            
        Returns:
            Dictionary with measurements and metadata
        """
        result = {
            'filename': Path(image_path).name,
            'success': False
        }
        
        # Load image to get dimensions
        image = cv2.imread(str(image_path))
        if image is None:
            result['error'] = 'Failed to load image'
            return result
        
        result['image_width'] = image.shape[1]
        result['image_height'] = image.shape[0]
        
        # Check calibration
        if not self.manual_calibration:
            result['error'] = 'No calibration provided'
            return result
        
        result['calibration_method'] = 'manual'
        result['pixels_per_cm'] = round(self.manual_calibration, 2)
        
        # Run YOLO detection
        results = self.model.predict(
            source=image_path,
            conf=conf_threshold,
            verbose=False
        )
        
        # Check if goat was detected
        if len(results) == 0 or results[0].masks is None:
            result['error'] = 'No goat detected'
            result['yolo_confidence'] = 0.0
            return result
        
        # Get first detection (highest confidence)
        detection = results[0]
        
        if len(detection.boxes) == 0:
            result['error'] = 'No goat detected'
            result['yolo_confidence'] = 0.0
            return result
        
        # Get confidence score
        confidence = float(detection.boxes[0].conf)
        result['yolo_confidence'] = round(confidence, 3)
        
        # Get segmentation mask
        mask = detection.masks[0].data[0].cpu().numpy()
        
        # Resize mask to original image size
        mask_resized = cv2.resize(
            mask,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
        mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
        
        # Extract measurements from mask
        measurements = self.extract_measurements(mask_binary, self.manual_calibration)
        
        if not measurements:
            result['error'] = 'Failed to extract measurements from mask'
            return result
        
        result.update(measurements)
        result['success'] = True
        
        # Remove debug points from output (only used for visualization)
        if '_debug_points' in result:
            del result['_debug_points']
        
        # Generate debug image if requested
        if self.debug:
            debug_img = image.copy()
            
            # Get bounding box from mask contour for visualization
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(contour)
                
                # Draw mask overlay (semi-transparent blue)
                overlay = image.copy()
                overlay[mask_binary > 0] = [255, 0, 0]  # Blue
                debug_img = cv2.addWeighted(debug_img, 0.7, overlay, 0.3, 0)
                
                # Draw bounding box
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 255), 3)
                
                # Draw vertical dividers for three sections
                third = w // 3
                cv2.line(debug_img, (x + third, y), (x + third, y + h), (0, 255, 0), 2)
                cv2.line(debug_img, (x + 2*third, y), (x + 2*third, y + h), (0, 255, 0), 2)
                
                # Draw horizontal ground line
                ground_level = y + h
                cv2.line(debug_img, (x, ground_level), (x + w, ground_level), (0, 165, 255), 3)
                cv2.putText(debug_img, "GROUND", (x + 10, ground_level + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
                
                # Draw measurement lines for each section
                if '_debug_points' in measurements:
                    debug_pts = measurements['_debug_points']
                    
                    # Head measurement line (RED)
                    if 'head' in debug_pts:
                        pt = debug_pts['head']
                        # Draw vertical line from top to ground
                        cv2.line(debug_img, (pt['x'], pt['top']), (pt['x'], pt['bottom']), 
                                (0, 0, 255), 3)
                        # Draw horizontal line across entire first third at the top point
                        cv2.line(debug_img, (x, pt['top']), (x + third, pt['top']), 
                                (0, 0, 255), 3)
                        # Draw horizontal markers at bottom
                        cv2.line(debug_img, (pt['x']-20, pt['bottom']), (pt['x']+20, pt['bottom']), 
                                (0, 0, 255), 3)
                        # Label with measurement in cm
                        cv2.putText(debug_img, f"{measurements['head_height_cm']}cm", 
                                   (pt['x'] + 25, (pt['top'] + pt['bottom']) // 2),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    # Withers measurement line (GREEN)
                    if 'withers' in debug_pts:
                        pt = debug_pts['withers']
                        cv2.line(debug_img, (pt['x'], pt['top']), (pt['x'], pt['bottom']), 
                                (0, 255, 0), 3)
                        # Draw horizontal line across entire middle third at the top point
                        cv2.line(debug_img, (x + third, pt['top']), (x + 2*third, pt['top']), 
                                (0, 255, 0), 3)
                        cv2.line(debug_img, (pt['x']-20, pt['bottom']), (pt['x']+20, pt['bottom']), 
                                (0, 255, 0), 3)
                        cv2.putText(debug_img, f"{measurements['withers_height_cm']}cm", 
                                   (pt['x'] + 25, (pt['top'] + pt['bottom']) // 2),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # Rump measurement line (CYAN)
                    if 'rump' in debug_pts:
                        pt = debug_pts['rump']
                        cv2.line(debug_img, (pt['x'], pt['top']), (pt['x'], pt['bottom']), 
                                (255, 255, 0), 3)
                        # Draw horizontal line across entire last third at the top point
                        cv2.line(debug_img, (x + 2*third, pt['top']), (x + w, pt['top']), 
                                (255, 255, 0), 3)
                        cv2.line(debug_img, (pt['x']-20, pt['bottom']), (pt['x']+20, pt['bottom']), 
                                (255, 255, 0), 3)
                        cv2.putText(debug_img, f"{measurements['rump_height_cm']}cm", 
                                   (pt['x'] + 25, (pt['top'] + pt['bottom']) // 2),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                
                # Add overall measurements at top
                text = f"Length: {measurements['body_length_cm']}cm"
                cv2.putText(debug_img, text, (x, y - 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
                
                heights_text = f"Heights - Head:{measurements['head_height_cm']} Withers:{measurements['withers_height_cm']} Rump:{measurements['rump_height_cm']}cm"
                cv2.putText(debug_img, heights_text, (x, y - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
                
                # Add confidence score
                conf_text = f"Confidence: {confidence:.2f}"
                cv2.putText(debug_img, conf_text, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
            
            # Create debug directory if it doesn't exist
            debug_dir = Path("debug")
            debug_dir.mkdir(exist_ok=True)
            
            output_path = debug_dir / f"debug_{Path(image_path).name}"
            cv2.imwrite(str(output_path), debug_img)
            result['debug_image'] = str(output_path)
        
        return result
    
    def process_batch(self, image_dir: str, output_file: str = 'side_yolo_measurements.json',
                     conf_threshold: float = 0.1):
        """
        Process all images in a directory
        
        Args:
            image_dir: Directory containing images
            output_file: Output JSON file path
            conf_threshold: YOLO confidence threshold
        """
        image_dir = Path(image_dir)
        results = []
        
        # Find all images
        image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.JPG"))
        
        if len(image_paths) == 0:
            print(f"No images found in {image_dir}")
            return
        
        print(f"Processing {len(image_paths)} images...")
        print(f"Using YOLO confidence threshold: {conf_threshold}")
        
        if self.debug:
            debug_dir = Path("debug")
            debug_dir.mkdir(exist_ok=True)
            print(f"Debug images will be saved to: {debug_dir.absolute()}")
        
        successful = 0
        
        for i, img_path in enumerate(image_paths, 1):
            print(f"[{i}/{len(image_paths)}] Processing {img_path.name}...", end=' ')
            
            result = self.process_image(img_path, conf_threshold)
            results.append(result)
            
            if result['success']:
                successful += 1
                conf = result['yolo_confidence']
                length = result['body_length_cm']
                head = result['head_height_cm']
                withers = result['withers_height_cm']
                rump = result['rump_height_cm']
                print(f"✓ L:{length}cm H:{head}/{withers}/{rump}cm (conf: {conf:.2f})")
            else:
                print(f"✗ {result.get('error', 'Unknown error')}")
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Results saved to {output_file}")
        print(f"{'='*60}")
        print(f"Total images: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {len(results) - successful}")
        
        if successful > 0:
            # Statistics
            successful_results = [r for r in results if r['success']]
            
            lengths = [r['body_length_cm'] for r in successful_results]
            head_heights = [r['head_height_cm'] for r in successful_results]
            withers_heights = [r['withers_height_cm'] for r in successful_results]
            rump_heights = [r['rump_height_cm'] for r in successful_results]
            confidences = [r['yolo_confidence'] for r in successful_results]
            
            print(f"\nMeasurement statistics:")
            print(f"  Body length: {min(lengths):.1f}cm - {max(lengths):.1f}cm (avg: {np.mean(lengths):.1f}cm)")
            print(f"  Head height: {min(head_heights):.1f}cm - {max(head_heights):.1f}cm (avg: {np.mean(head_heights):.1f}cm)")
            print(f"  Withers height: {min(withers_heights):.1f}cm - {max(withers_heights):.1f}cm (avg: {np.mean(withers_heights):.1f}cm)")
            print(f"  Rump height: {min(rump_heights):.1f}cm - {max(rump_heights):.1f}cm (avg: {np.mean(rump_heights):.1f}cm)")
            print(f"  YOLO confidence: {min(confidences):.2f} - {max(confidences):.2f} (avg: {np.mean(confidences):.2f})")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO-based Goat Measurements')
    parser.add_argument('--model', required=True,
                       help='Path to trained YOLO model (.pt file)')
    parser.add_argument('--calibration', required=True,
                       help='Calibration JSON file from calibration_tool.py')
    parser.add_argument('--image', help='Single image to process')
    parser.add_argument('--batch', help='Directory of images to process')
    parser.add_argument('--output', default='side_yolo_measurements.json',
                       help='Output JSON file')
    parser.add_argument('--conf', type=float, default=0.1,
                       help='YOLO confidence threshold (default: 0.1)')
    parser.add_argument('--debug', action='store_true',
                       help='Save debug images with visualizations')
    
    args = parser.parse_args()
    
    # Load calibration
    with open(args.calibration, 'r') as f:
        cal_data = json.load(f)
        pixels_per_cm = cal_data['pixels_per_cm']
    
    print(f"Using calibration: {pixels_per_cm:.2f} pixels/cm")
    print(f"Using YOLO model: {args.model}")
    
    # Initialize extractor
    extractor = YOLOGoatMeasurements(args.model, pixels_per_cm)
    extractor.debug = args.debug
    
    if args.image:
        # Single image
        result = extractor.process_image(args.image, args.conf)
        print(json.dumps(result, indent=2))
    
    elif args.batch:
        # Batch processing
        extractor.process_batch(args.batch, args.output, args.conf)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()