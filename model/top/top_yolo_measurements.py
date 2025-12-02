"""
Hi! Please read :P - Ethan

NOTE - HOW I GOT THIS WORKING:
I manually traced goat outlines on 20 images using RoboFlow to create segmentation masks, then split them for training/validation. 
These are the usable top images that we got from Becky on google photos. These are in /train and /val. 
Each of these has an /images and /labels subfolder. Images are the actual images, and labels are the data containing the segmentation masks in YOLO format.
I then trained YOLOv8-seg for epochs using these folders, which learned to recognize and segment goats from above by fine-tuning pretrained weights. 
Now when you run this using a new image, YOLO will output a segmentation mask of the goat in the image using its weights from these dirs.

NOTE - MEASUREMENT CALIBRATION: 
Uses chute dimensions for calibration instead of manual markers:
- Chute internal width: 40.64 cm (16 inches)
- Chute internal length: 121.92 cm (48 inches)
The script detects the chute edges (green metal bars on left/right) and uses the known width for calibration.

NOTE - OUTPUTS:
Outputs a top_yolo_measurements.json with measurements for each image processed in the following format:

    top_yolo_measurements.json
        {
            "filename": "IMG_xxx.jpg",
            "success": true,
            "image_width": 4080,
            "image_height": 3072,
            "calibration_method": "chute_width",
            "pixels_per_cm": 25.4,
            "yolo_confidence": 0.45,
            "body_length_cm": 95.3,
            "body_width_front_cm": 28.5,
            "body_width_mid_cm": 32.1,
            "body_width_rear_cm": 30.8,
            "body_area_square_cm": 2850.2,
            "length_to_width_ratio": 2.97,
            "debug_image": "debug/debug_IMG_xxx.jpg"
        }

Also outputs a debug folder with images showing the segmentation mask and measurement lines drawn on the image for visual verification.

Currently, with such a small dataset, we cannot expect high confidence scores. 
They do a fine job for now, but we can expect dramatic improvements with more training data.

"""

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
    
    def extract_measurements(self, mask):
        """
        mask: binary segmentation mask
        returns: { max_height_pixels, max_height_col }
        """
        mask = (mask > 0).astype(np.uint8)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None

        cnt = max(cnts, key=cv2.contourArea)

        h, w = mask.shape
        contour_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(contour_mask, [cnt], -1, 255, -1)

        max_height = 0
        max_col_x = 0

        # *** CHANGED: scan vertical (top→bottom) instead of horizontal ***
        for x in range(w):
            ys = np.where(contour_mask[:, x] == 255)[0]
            if ys.size > 0:
                height = ys[-1] - ys[0]
                if height > max_height:
                    max_height = height
                    max_col_x = x

        return {
            "max_height_pixels": int(max_height),
            "max_height_col": int(max_col_x)
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
        
        confidence = float(detection.boxes[0].conf)
        result['yolo_confidence'] = round(confidence, 3)
        
        mask = detection.masks[0].data[0].cpu().numpy()
        
        mask_resized = cv2.resize(
            mask,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
        mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
        
        measurements = self.extract_measurements(mask_binary)
        
        if not measurements:
            result['error'] = 'Failed to extract measurements from mask'
            return result
        
        result.update(measurements)
        result['success'] = True
        
        if '_debug_points' in result:
            del result['_debug_points']
        
        # ============================
        # DEBUG IMAGE (OLD STYLE)
        # ============================
        if self.debug:
            debug_img = image.copy()

            # mask overlay
            overlay = image.copy()
            overlay[mask_binary > 0] = [255, 0, 0]
            debug_img = cv2.addWeighted(debug_img, 0.7, overlay, 0.3, 0)

            # contour
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = max(contours, key=cv2.contourArea)
                cv2.drawContours(debug_img, [contour], -1, (0, 255, 255), 3)

            # *** CHANGED: draw VERTICAL measurement line ***
            x = measurements["max_height_col"]
            max_h = measurements["max_height_pixels"]

            ys = np.where(mask_binary[:, x] == 255)[0]
            if ys.size > 0:
                y1 = int(ys[0])
                y2 = int(ys[-1])

                # vertical line like OLD version
                cv2.line(debug_img, (x, y1), (x, y2), (0, 0, 255), 4)

                # label text
                cv2.putText(
                    debug_img,
                    f"Height(px): {max_h}",
                    (x + 10, (y1 + y2) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            debug_dir = Path("debug")
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
                print(f"✓ (conf: {result['yolo_confidence']:.2f})")
            else:
                print(f"✗ {result.get('error', 'Unknown error')}")
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nSaved to {output_file}")
        print(f"Successful: {successful}")
        print(f"Failed: {len(results) - successful}")

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
