'''
USAGE:
    cd model/top

    python top_yolo_measurements.py \
    --model best.pt \
    --calibration top_calibration.json \
    --batch ../images/top/ \
    --debug

yolo based measurements for top images of goats
uses top_calibration.json for cm to pixels conversion. this will eventually be hardcoded once cameras are fixed.
using model top/best.pt

OUTPUTS: top_yolo_measurements.json in the following format for each goat:

        {
            "filename": "IMG_20251113_093527751_HDR_jpg.rf.a0b44d374199251ffd01c478f468ce91.jpg",
            "success": true,
            "image_width": 4080,
            "image_height": 3072,
            "calibration_method": "manual",
            "pixels_per_cm": 25.73,
            "yolo_confidence": 0.785,
            "max_width_pixels": 845,
            "max_width_col": 1613,
            "body_width_cm": 32.84,
            "body_axis_angle": 2.5,
            "body_area_square_cm": 1435.25,
            "debug_image": "debug/debug_IMG_20251113_093527751_HDR_jpg.rf.a0b44d374199251ffd01c478f468ce91.jpg"
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

class YOLOGoatMeasurementsTop:
    def __init__(self, model_path: str, pixels_per_cm: Optional[float] = None):
        self.model = YOLO(model_path)
        self.manual_calibration = pixels_per_cm
        self.debug = False
    
    def compute_body_axis(self, mask):
        """
        Compute the principal axis of the body mask using PCA.
        Returns: dict with angle (degrees), center point, direction vector, and eigenvalues
        """
        # Get all mask pixels
        y_coords, x_coords = np.where(mask > 0)
        
        if len(x_coords) < 10:
            return None
        
        # Stack into Nx2 array
        points = np.column_stack([x_coords, y_coords])
        
        # Compute PCA
        mean = points.mean(axis=0)
        centered = points - mean
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        # Principal axis is the eigenvector with largest eigenvalue
        principal_idx = np.argmax(eigenvalues)
        principal_axis = eigenvectors[:, principal_idx]
        
        # Angle in degrees (from horizontal)
        angle = np.degrees(np.arctan2(principal_axis[1], principal_axis[0]))
        
        return {
            'angle': float(angle),
            'center': mean.astype(float),
            'direction': principal_axis.astype(float),
            'eigenvalues': eigenvalues.astype(float)
        }
    
    def extract_measurements(self, mask, pixels_per_cm):
        """
        mask: binary segmentation mask
        pixels_per_cm: calibration value
        returns: { body_width_cm, body_area_square_cm, measurement details }
        
        Uses PCA to find body axis, then measures width PERPENDICULAR to that axis.
        Finds maximum body width in TORSO region (excludes head/neck automatically)
        """
        mask = (mask > 0).astype(np.uint8)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None

        cnt = max(cnts, key=cv2.contourArea)

        h, w = mask.shape
        contour_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(contour_mask, [cnt], -1, 255, -1)

        # Compute body axis using PCA
        axis_info = self.compute_body_axis(contour_mask)
        
        if not axis_info:
            return None
        
        # Rotate mask so body axis is horizontal
        center = (float(axis_info['center'][0]), float(axis_info['center'][1]))
        angle = axis_info['angle']
        
        # Create rotation matrix to align body horizontally
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_mask = cv2.warpAffine(contour_mask, rotation_matrix, (w, h))

        # Get bounding box of rotated mask
        x, y, bbox_w, bbox_h = cv2.boundingRect(rotated_mask)

        # SCAN ALONG BODY AXIS (now horizontal after rotation) to build width profile
        # Width is now perpendicular to body axis
        width_profile = []
        for col_x in range(x, x + bbox_w):
            if col_x >= w:
                continue
            ys = np.where(rotated_mask[:, col_x] == 255)[0]
            if ys.size > 0:
                width = ys[-1] - ys[0]  # Perpendicular distance
                width_profile.append((col_x, width))
            else:
                width_profile.append((col_x, 0))

        if len(width_profile) == 0:
            return None

        # Find where body stabilizes into torso (after neck)
        widths = [w for _, w in width_profile]
        
        # Use a rolling window to find where width stabilizes
        window_size = max(5, len(widths) // 20)
        
        # Find the point where we reach ~75% of max width and stay there
        max_width_overall = max(widths)
        torso_threshold = max_width_overall * 0.75
        
        # Find first point where we exceed threshold and stay above it
        torso_start_idx = 0
        for i in range(len(widths) - window_size):
            window = widths[i:i+window_size]
            if sum(1 for w in window if w >= torso_threshold) >= window_size * 0.7:
                torso_start_idx = i
                break
        
        # Find last point where we're above threshold
        torso_end_idx = len(widths) - 1
        for i in range(len(widths) - 1, window_size, -1):
            window = widths[i-window_size:i]
            if sum(1 for w in window if w >= torso_threshold) >= window_size * 0.7:
                torso_end_idx = i
                break
        
        # Find maximum width in the torso region only
        max_width = 0
        max_col_x = 0
        for i in range(torso_start_idx, torso_end_idx + 1):
            col_x, width = width_profile[i]
            if width > max_width:
                max_width = width
                max_col_x = col_x

        # Calculate body area
        area_pixels = cv2.contourArea(cnt)
        area_square_cm = area_pixels / (pixels_per_cm ** 2)

        # Store scan region for visualization (in rotated coordinates)
        scan_start = width_profile[torso_start_idx][0] if torso_start_idx < len(width_profile) else x
        scan_end = width_profile[torso_end_idx][0] if torso_end_idx < len(width_profile) else x + bbox_w

        # Transform measurement point back to original coordinates
        # max_col_x is in rotated space, need to find where measurement line intersects in original
        ys_rotated = np.where(rotated_mask[:, max_col_x] == 255)[0]
        if ys_rotated.size > 0:
            y1_rot = ys_rotated[0]
            y2_rot = ys_rotated[-1]
            
            # Transform back to original coordinates
            inv_rotation = cv2.invertAffineTransform(rotation_matrix)
            pt1_orig = np.dot(inv_rotation, np.array([max_col_x, y1_rot, 1]))
            pt2_orig = np.dot(inv_rotation, np.array([max_col_x, y2_rot, 1]))
            
            measurement_line_original = (pt1_orig, pt2_orig)
        else:
            measurement_line_original = None

        return {
            "max_width_pixels": int(max_width),
            "max_width_col": int(max_col_x),
            "body_width_cm": round(max_width / pixels_per_cm, 2),
            "body_axis_angle": round(angle, 2),
            "body_area_square_cm": round(area_square_cm, 2),
            "_debug_scan_range": (scan_start, scan_end),
            "_debug_bbox": (x, y, bbox_w, bbox_h),
            "_debug_torso_region": (torso_start_idx, torso_end_idx, len(width_profile)),
            "_debug_axis_info": axis_info,
            "_debug_rotation_matrix": rotation_matrix,
            "_debug_rotated_mask": rotated_mask,
            "_debug_measurement_line": measurement_line_original
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
            result['error'] = 'No segmentation masks returned by YOLO'
            result['yolo_confidence'] = 0.0
            return result

        detection = results[0]

        if len(detection.boxes) == 0:
            n_masks = len(detection.masks) if detection.masks is not None else 0
            result['error'] = f'YOLO returned {n_masks} masks but no bounding boxes (image: {image_path})'
            result['yolo_confidence'] = 0.0
            return result

        # With 2-class model, we have separate body and head masks
        # Class 0 = goat_body, Class 1 = goat_head
        body_mask = None
        head_mask = None
        body_confidence = 0.0
        detected_classes = []

        for i, box in enumerate(detection.boxes):
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            detected_classes.append((class_id, conf))

            if class_id == 0:  # goat_body
                body_mask = detection.masks[i].data[0].cpu().numpy()
                body_confidence = conf
            elif class_id == 1:  # goat_head
                head_mask = detection.masks[i].data[0].cpu().numpy()

        if body_mask is None:
            # Show what was detected instead
            class_info = ', '.join([f'class {c} ({conf:.2f})' for c, conf in detected_classes])
            result['error'] = f'No body (class 0) detected. Found: {class_info}'
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
        
        # Store debug info before updating result
        debug_axis_info = measurements.get('_debug_axis_info')
        debug_rotation_matrix = measurements.get('_debug_rotation_matrix')
        debug_rotated_mask = measurements.get('_debug_rotated_mask')
        debug_measurement_line = measurements.get('_debug_measurement_line')
        debug_scan_range = measurements.get('_debug_scan_range')
        debug_bbox = measurements.get('_debug_bbox')
        
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

            # Draw body axis line using PCA results
            if debug_axis_info:
                center = debug_axis_info['center'].astype(int)
                direction = debug_axis_info['direction']
                angle = debug_axis_info['angle']
                
                # Draw axis line through center (extend 40% of body length in each direction)
                if contours:
                    contour = max(contours, key=cv2.contourArea)
                    bx, by, bw, bh = cv2.boundingRect(contour)
                    axis_length = int(max(bw, bh) * 0.4)
                    
                    # Calculate axis endpoints
                    axis_start = (
                        int(center[0] - direction[0] * axis_length),
                        int(center[1] - direction[1] * axis_length)
                    )
                    axis_end = (
                        int(center[0] + direction[0] * axis_length),
                        int(center[1] + direction[1] * axis_length)
                    )
                    
                    # Draw PCA-computed body axis (green, thick)
                    cv2.line(debug_img, axis_start, axis_end, (0, 255, 0), 4)
                    cv2.putText(debug_img, f"Body Axis ({angle:.1f}deg)", 
                               (axis_start[0], axis_start[1] - 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Draw measurement line perpendicular to body axis
            if debug_measurement_line:
                pt1, pt2 = debug_measurement_line
                pt1_int = tuple(pt1.astype(int))
                pt2_int = tuple(pt2.astype(int))
                
                # Draw perpendicular measurement line (red, thick)
                cv2.line(debug_img, pt1_int, pt2_int, (0, 0, 255), 4)
                
                # Add caps on measurement line
                cv2.circle(debug_img, pt1_int, 8, (0, 0, 255), -1)
                cv2.circle(debug_img, pt2_int, 8, (0, 0, 255), -1)
                
                # Label with cm measurement
                mid_point = ((pt1_int[0] + pt2_int[0]) // 2, (pt1_int[1] + pt2_int[1]) // 2)
                cv2.putText(
                    debug_img,
                    f"{result['body_width_cm']}cm",
                    (mid_point[0] + 20, mid_point[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3
                )

            # Add summary text
            summary_y = 60
            cv2.putText(debug_img, f"Max Width: {result['body_width_cm']}cm", 
                       (10, summary_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(debug_img, f"Axis Angle: {result.get('body_axis_angle', 0):.1f}deg", 
                       (10, summary_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(debug_img, f"Body Conf: {body_confidence:.2f}", 
                       (10, summary_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            # Legend
            legend_y = debug_img.shape[0] - 60
            cv2.putText(debug_img, "Blue=Body, Green=Head+Axis, Red=Width", 
                       (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            debug_dir = Path(__file__).parent / 'debug'
            debug_dir.mkdir(exist_ok=True)
            output_path = debug_dir / f"debug_{Path(image_path).name}"
            cv2.imwrite(str(output_path), debug_img)
            result['debug_image'] = str(output_path)

        
        return result
    
    def process_batch(self, image_dir: str, output_file: str = 'top_yolo_measurements.json',
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
                angle = result.get('body_axis_angle', 0)
                print(f"✓ W:{width}cm, Angle:{angle:.1f}° (conf: {result['yolo_confidence']:.2f})")
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
            angles = [r.get('body_axis_angle', 0) for r in successful_results]
            confidences = [r['yolo_confidence'] for r in successful_results]
            
            print(f"\nMeasurement statistics:")
            print(f"  Body width: {min(widths):.1f}cm - {max(widths):.1f}cm (avg: {np.mean(widths):.1f}cm)")
            print(f"  Axis angle: {min(angles):.1f}° - {max(angles):.1f}° (avg: {np.mean(angles):.1f}°)")
            print(f"  YOLO confidence: {min(confidences):.2f} - {max(confidences):.2f} (avg: {np.mean(confidences):.2f})")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO-based Goat Measurements (Top View with PCA)')
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
    
    extractor = YOLOGoatMeasurementsTop(args.model, pixels_per_cm)
    extractor.debug = args.debug
    
    if args.image:
        print(json.dumps(extractor.process_image(args.image, args.conf), indent=2))
    
    elif args.batch:
        extractor.process_batch(args.batch, args.output, args.conf)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()