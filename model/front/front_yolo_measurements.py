'''
FIND USAGE IN MODEL-README.md

yolo based measurements for side images of goats
uses front_calibration.json for cm to pixels conversion. this will eventually be hardcoded once cameras are fixed.
using model front/best.pt

OUTPUTS: front_yolo_measurements.json in the following format for each goat:

        {
            "filename": "IMG_20251113_094244496_HDR_jpg.rf.ad847b41e49ef6ebba4664c6c8a690e0.jpg",
            "success": true,
            "image_width": 4080,
            "image_height": 3072,
            "calibration_method": "manual",
            "pixels_per_cm": 22.07,
            "yolo_confidence": 0.916,
            "max_width_pixels": 815,
            "max_width_row": 1152,
            "body_width_cm": 36.94,
            "body_area_square_cm": 1608.07,
            "debug_image": "debug/debug_IMG_20251113_094244496_HDR_jpg.rf.ad847b41e49ef6ebba4664c6c8a690e0.jpg"
        }

OUTPUTS: debug/ folder with images showing segmentation mask and measurement lines (if --debug flag is used)
'''

import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
from ultralytics import YOLO

# Chute dimensions from manufacturer specs (Lakeland PN-440)
CHUTE_WIDTH_CM = 40.64  # 16 inches internal width
CHUTE_LENGTH_CM = 121.92  # 48 inches internal length

class YOLOGoatMeasurementsFront:
    def __init__(self, model_path: str, pixels_per_cm: Optional[float] = None):
        self.model = YOLO(model_path)
        self.manual_calibration = pixels_per_cm
        self.debug = False
    
    def extract_measurements(self, mask, pixels_per_cm):
        """
        mask: binary segmentation mask (front view - goat facing camera)
        pixels_per_cm: calibration value
        returns: { body_width_cm, body_area_square_cm, measurement details }
        
        Finds maximum body width (chest/shoulder) in TORSO region
        Front view: body is vertical, measure horizontal width (left-right)
        """
        mask = (mask > 0).astype(np.uint8)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None

        cnt = max(cnts, key=cv2.contourArea)

        h, w = mask.shape
        contour_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(contour_mask, [cnt], -1, 255, -1)

        # Get bounding box
        x, y, bbox_w, bbox_h = cv2.boundingRect(cnt)

        # Front view: scan ROWS (y-axis) to build width profile
        # Body is vertical (head at top, legs at bottom)
        width_profile = []
        for row_y in range(y, y + bbox_h):
            if row_y >= h:
                continue
            xs = np.where(contour_mask[row_y, :] == 255)[0]
            if xs.size > 0:
                width = xs[-1] - xs[0]  # Left-right extent
                width_profile.append((row_y, width))
            else:
                width_profile.append((row_y, 0))

        if len(width_profile) == 0:
            return None

        # Find where body stabilizes into torso (chest/shoulder area)
        widths = [w for _, w in width_profile]
        
        # Use rolling window
        window_size = max(5, len(widths) // 20)
        
        # Find torso region (75% of max width)
        max_width_overall = max(widths)
        torso_threshold = max_width_overall * 0.75
        
        # Find first point where we exceed threshold
        torso_start_idx = 0
        for i in range(len(widths) - window_size):
            window = widths[i:i+window_size]
            if sum(1 for w in window if w >= torso_threshold) >= window_size * 0.7:
                torso_start_idx = i
                break
        
        # Find last point above threshold
        torso_end_idx = len(widths) - 1
        for i in range(len(widths) - 1, window_size, -1):
            window = widths[i-window_size:i]
            if sum(1 for w in window if w >= torso_threshold) >= window_size * 0.7:
                torso_end_idx = i
                break
        
        # Find maximum width in torso region
        max_width = 0
        max_row_y = 0
        for i in range(torso_start_idx, torso_end_idx + 1):
            row_y, width = width_profile[i]
            if width > max_width:
                max_width = width
                max_row_y = row_y

        # Calculate body area
        area_pixels = cv2.contourArea(cnt)
        area_square_cm = area_pixels / (pixels_per_cm ** 2)

        # Store scan region for visualization
        scan_start = width_profile[torso_start_idx][0] if torso_start_idx < len(width_profile) else y
        scan_end = width_profile[torso_end_idx][0] if torso_end_idx < len(width_profile) else y + bbox_h

        return {
            "max_width_pixels": int(max_width),
            "max_width_row": int(max_row_y),
            "body_width_cm": round(max_width / pixels_per_cm, 2),
            "body_area_square_cm": round(area_square_cm, 2),
            "_debug_scan_range": (scan_start, scan_end),
            "_debug_bbox": (x, y, bbox_w, bbox_h),
            "_debug_torso_region": (torso_start_idx, torso_end_idx, len(width_profile))
        }
    
    def process_image(self, image_path: str, conf_threshold: float = 0.1) -> Dict:
        result = {
            'filename': Path(image_path).name,
            'success': False
        }
        
        image = cv2.imread(str(image_path))
        if image is None:
            result['error'] = 'Failed to load image'
            return result
        
        result['image_width'] = image.shape[1]
        result['image_height'] = image.shape[0]
        
        if not self.manual_calibration:
            result['error'] = 'No calibration provided'
            return result
        
        result['calibration_method'] = 'manual'
        result['pixels_per_cm'] = round(self.manual_calibration, 2)
        
        results = self.model.predict(
            source=image_path,
            conf=conf_threshold,
            verbose=False
        )
        
        if len(results) == 0 or results[0].masks is None:
            result['error'] = 'No goat detected'
            result['yolo_confidence'] = 0.0
            return result
        
        detection = results[0]
        
        if len(detection.boxes) == 0:
            result['error'] = 'No goat detected'
            result['yolo_confidence'] = 0.0
            return result
        
        # With 2-class model, we have separate body and head masks
        # Class 0 = goat_body, Class 1 = goat_head
        body_mask = None
        head_mask = None
        body_confidence = 0.0
        
        for i, box in enumerate(detection.boxes):
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            if class_id == 0:  # goat_body
                body_mask = detection.masks[i].data[0].cpu().numpy()
                body_confidence = conf
            elif class_id == 1:  # goat_head
                head_mask = detection.masks[i].data[0].cpu().numpy()
        
        if body_mask is None:
            result['error'] = 'No body mask detected'
            result['yolo_confidence'] = 0.0
            return result
        
        result['yolo_confidence'] = round(body_confidence, 3)
        
        # Resize body mask to original image size
        body_mask_resized = cv2.resize(
            body_mask,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
        mask_binary = (body_mask_resized > 0.5).astype(np.uint8) * 255
        
        # Optional: also resize head mask for visualization
        head_mask_binary = None
        if head_mask is not None:
            head_mask_resized = cv2.resize(
                head_mask,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
            head_mask_binary = (head_mask_resized > 0.5).astype(np.uint8) * 255
        
        measurements = self.extract_measurements(mask_binary, self.manual_calibration)
        
        if not measurements:
            result['error'] = 'Failed to extract measurements from mask'
            return result
        
        result.update(measurements)
        result['success'] = True
        
        # Remove debug fields from JSON output
        debug_fields = [k for k in result.keys() if k.startswith('_debug')]
        for field in debug_fields:
            del result[field]
        

        if self.debug:
            debug_img = image.copy()

            # Body mask overlay (blue)
            overlay = image.copy()
            overlay[mask_binary > 0] = [255, 0, 0]
            debug_img = cv2.addWeighted(debug_img, 0.7, overlay, 0.3, 0)
            
            # Head mask overlay (green) if available
            if head_mask_binary is not None:
                overlay_head = debug_img.copy()
                overlay_head[head_mask_binary > 0] = [0, 255, 0]
                debug_img = cv2.addWeighted(debug_img, 0.7, overlay_head, 0.3, 0)

            # Body contour (yellow)
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = max(contours, key=cv2.contourArea)
                cv2.drawContours(debug_img, [contour], -1, (0, 255, 255), 3)
            
            # Head contour (cyan) if available
            if head_mask_binary is not None:
                head_contours, _ = cv2.findContours(head_mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if head_contours:
                    head_contour = max(head_contours, key=cv2.contourArea)
                    cv2.drawContours(debug_img, [head_contour], -1, (255, 255, 0), 3)

            # Draw scan region boundaries (torso detection region)
            if '_debug_scan_range' in measurements:
                scan_start, scan_end = measurements["_debug_scan_range"]
                img_w = debug_img.shape[1]
                # Horizontal lines showing detected torso region
                cv2.line(debug_img, (0, scan_start), (img_w, scan_start), (255, 0, 255), 2)
                cv2.line(debug_img, (0, scan_end), (img_w, scan_end), (255, 0, 255), 2)
                cv2.putText(debug_img, "Torso Region", (10, scan_start - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

            # Draw maximum width measurement line
            y = measurements["max_width_row"]
            max_w = measurements["max_width_pixels"]

            xs = np.where(mask_binary[y, :] == 255)[0]
            if xs.size > 0:
                x1 = int(xs[0])
                x2 = int(xs[-1])

                # Horizontal measurement line
                cv2.line(debug_img, (x1, y), (x2, y), (0, 0, 255), 4)
                
                # Vertical caps on measurement line
                cv2.line(debug_img, (x1, y-20), (x1, y+20), (0, 0, 255), 3)
                cv2.line(debug_img, (x2, y-20), (x2, y+20), (0, 0, 255), 3)

                # Label with cm measurement
                cv2.putText(
                    debug_img,
                    f"{measurements['body_width_cm']}cm",
                    ((x1 + x2) // 2 - 50, y - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2
                )

            # Add summary text
            summary_y = 60
            cv2.putText(debug_img, f"Max Width: {measurements['body_width_cm']}cm", 
                       (10, summary_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(debug_img, f"Body Conf: {body_confidence:.2f}", 
                       (10, summary_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            # Legend
            legend_y = debug_img.shape[0] - 60
            cv2.putText(debug_img, "Blue=Body, Green=Head", 
                       (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            debug_dir = Path(__file__).parent / 'debug'  # front/debug/ (relative to this script)
            debug_dir.mkdir(exist_ok=True)
            output_path = debug_dir / f"debug_{Path(image_path).name}"
            cv2.imwrite(str(output_path), debug_img)
            result['debug_image'] = str(output_path)

        
        return result
    
    def process_batch(self, image_dir: str, output_file: str = 'front_yolo_measurements.json',
                     conf_threshold: float = 0.1):
        image_dir = Path(image_dir)
        results = []
        
        image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.JPG"))
        
        if len(image_paths) == 0:
            print(f"No images found in {image_dir}")
            return
        
        print(f"Processing {len(image_paths)} images...")
        print(f"Using YOLO confidence threshold: {conf_threshold}")
        print(f"Using manual calibration: {self.manual_calibration:.2f} pixels/cm")
        
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
                width = result['body_width_cm']
                print(f"✓ W:{width}cm (conf: {result['yolo_confidence']:.2f})")
            else:
                print(f"✗ {result.get('error', 'Unknown error')}")
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Results saved to {output_file}")
        print(f"{'='*60}")
        print(f"Total images: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {len(results) - successful}")
        
        if successful > 0:
            successful_results = [r for r in results if r['success']]
            
            widths = [r['body_width_cm'] for r in successful_results]
            confidences = [r['yolo_confidence'] for r in successful_results]
            
            print(f"\nMeasurement statistics:")
            print(f"  Body width: {min(widths):.1f}cm - {max(widths):.1f}cm (avg: {np.mean(widths):.1f}cm)")
            print(f"  YOLO confidence: {min(confidences):.2f} - {max(confidences):.2f} (avg: {np.mean(confidences):.2f})")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO-based Goat Measurements (Top View)')
    parser.add_argument('--model', required=True,
                       help='Path to trained YOLO model (.pt file)')
    parser.add_argument('--calibration', required=True,
                       help='Calibration JSON file')
    parser.add_argument('--image', help='Single image to process')
    parser.add_argument('--batch', help='Directory of images to process')
    parser.add_argument('--output', default='top_yolo_measurements.json')
    parser.add_argument('--conf', type=float, default=0.1)
    parser.add_argument('--debug', action='store_true')
    
    args = parser.parse_args()
    
    with open(args.calibration, 'r') as f:
        cal_data = json.load(f)
        pixels_per_cm = cal_data['pixels_per_cm']
    
    print(f"Using calibration: {pixels_per_cm:.2f} pixels/cm")
    
    extractor = YOLOGoatMeasurementsFront(args.model, pixels_per_cm)
    extractor.debug = args.debug
    
    if args.image:
        print(json.dumps(extractor.process_image(args.image, args.conf), indent=2))
    
    elif args.batch:
        extractor.process_batch(args.batch, args.output, args.conf)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()