"""
Manual Calibration Tool for Top View - Floor Grating Width

This tool allows you to manually click on the left and right edges of the floor grating
across multiple images, then calculates the average pixels_per_cm value.

The floor grating spans the internal chute width: 40.64 cm (16 inches)

Usage:
    a few specific images:
    python top_calibration_tool.py image1.jpg image2.jpg image3.jpg ...
    
    Or process entire top pics folder (recommended):
    python top_calibration_tool.py ../pictures/top/*.jpg

Instructions:
    1. Click on the LEFT edge of the floor grating
    2. Click on the RIGHT edge of the floor grating
    3. Repeat for each image
    4. Script calculates average pixels_per_cm and saves to top_calibration.json
"""

import cv2
import numpy as np
import json
import sys
from pathlib import Path

# Known grating width (internal chute width)
GRATING_WIDTH_CM = 40.64  # 16 inches

class TopCalibrationTool:
    def __init__(self):
        self.measurements = []
        self.current_image = None
        self.current_image_name = None
        self.points = []
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            
            # Draw point
            cv2.circle(self.current_image, (x, y), 8, (0, 0, 255), -1)
            
            if len(self.points) == 1:
                cv2.putText(self.current_image, "Top Edge", (x + 15, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            elif len(self.points) == 2:
                cv2.putText(self.current_image, "Bottom Edge", (x + 15, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Draw line between points
                cv2.line(self.current_image, self.points[0], self.points[1], 
                        (0, 255, 0), 3)
                
                # Calculate distance
                p1, p2 = self.points
                distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                pixels_per_cm = distance / GRATING_WIDTH_CM
                
                # Display measurement
                mid_x = (p1[0] + p2[0]) // 2
                mid_y = (p1[1] + p2[1]) // 2
                text = f"{distance:.1f}px = {GRATING_WIDTH_CM}cm"
                cv2.putText(self.current_image, text, (mid_x - 100, mid_y - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                text2 = f"{pixels_per_cm:.2f} pixels/cm"
                cv2.putText(self.current_image, text2, (mid_x - 100, mid_y + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Calibration', self.current_image)
    
    def calibrate_image(self, image_path: str):
        """Calibrate a single image"""
        print(f"\nProcessing: {Path(image_path).name}")
        
        # Load image
        original = cv2.imread(image_path)
        if original is None:
            print(f"  ✗ Failed to load image")
            return False
        
        self.current_image = original.copy()
        self.current_image_name = Path(image_path).name
        self.points = []
        
        # Display instructions
        instructions = [
            "Click on the TOP edge of the floor grating",
            "Then click on the BOTTOM edge",
            "Press SPACE when done, ESC to skip"
        ]
        
        img_with_text = self.current_image.copy()
        y_offset = 30
        for i, text in enumerate(instructions):
            cv2.putText(img_with_text, text, (10, y_offset + i*30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow('Calibration', img_with_text)
        cv2.setMouseCallback('Calibration', self.mouse_callback)
        
        # Wait for user input
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == 32:  # SPACE - confirm
                if len(self.points) == 2:
                    # Calculate measurement
                    p1, p2 = self.points
                    distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                    pixels_per_cm = distance / GRATING_WIDTH_CM
                    
                    self.measurements.append({
                        'image': self.current_image_name,
                        'top_point': self.points[0],
                        'bottom_point': self.points[1],
                        'pixel_distance': round(distance, 2),
                        'pixels_per_cm': round(pixels_per_cm, 4)
                    })
                    
                    print(f"  ✓ Measured: {distance:.1f} pixels = {GRATING_WIDTH_CM}cm")
                    print(f"  ✓ Pixels per cm: {pixels_per_cm:.4f}")
                    return True
                else:
                    print(f"  ✗ Need 2 points, only have {len(self.points)}")
                    
            elif key == 27:  # ESC - skip
                print(f"  ⊘ Skipped")
                return False
    
    def save_calibration(self, output_file: str = 'top_calibration.json'):
        """Save averaged calibration data"""
        if len(self.measurements) == 0:
            print("\nNo measurements collected!")
            return
        
        # Calculate statistics
        pixels_per_cm_values = [m['pixels_per_cm'] for m in self.measurements]
        avg_pixels_per_cm = np.mean(pixels_per_cm_values)
        std_pixels_per_cm = np.std(pixels_per_cm_values)
        min_pixels_per_cm = np.min(pixels_per_cm_values)
        max_pixels_per_cm = np.max(pixels_per_cm_values)
        
        # Calculate variation
        variation_percent = (std_pixels_per_cm / avg_pixels_per_cm) * 100
        
        # Prepare output
        calibration_data = {
            'method': 'manual_multi_image_average',
            'grating_width_cm': GRATING_WIDTH_CM,
            'num_images': len(self.measurements),
            'measurements': self.measurements,
            'pixels_per_cm': round(avg_pixels_per_cm, 4),
            'std_deviation': round(std_pixels_per_cm, 4),
            'variation_percent': round(variation_percent, 2),
            'min_pixels_per_cm': round(min_pixels_per_cm, 4),
            'max_pixels_per_cm': round(max_pixels_per_cm, 4)
        }
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Calibration saved to: {output_file}")
        print(f"{'='*60}")
        print(f"Images processed: {len(self.measurements)}")
        print(f"Average pixels/cm: {avg_pixels_per_cm:.4f}")
        print(f"Standard deviation: {std_pixels_per_cm:.4f}")
        print(f"Variation: {variation_percent:.2f}%")
        print(f"Range: {min_pixels_per_cm:.4f} - {max_pixels_per_cm:.4f}")
        
        if variation_percent > 5:
            print(f"\n⚠️  WARNING: High variation ({variation_percent:.1f}%)")
            print(f"   This suggests inconsistent clicking or camera position changes.")
            print(f"   Consider recalibrating for more accuracy.")
        else:
            print(f"\n✓ Good calibration! Low variation ({variation_percent:.1f}%)")

def main():
    if len(sys.argv) < 2:
        print("Usage: python top_calibration_tool.py image1.jpg image2.jpg ...")
        print("   Or: python top_calibration_tool.py images/*.jpg")
        sys.exit(1)
    
    image_paths = sys.argv[1:]
    
    print(f"{'='*60}")
    print(f"Top View Calibration Tool - Floor Grating Width")
    print(f"{'='*60}")
    print(f"Known grating width: {GRATING_WIDTH_CM}cm (16 inches)")
    print(f"Images to process: {len(image_paths)}")
    print(f"\nInstructions:")
    print(f"  1. Click TOP edge of floor grating")
    print(f"  2. Click BOTTOM edge of floor grating")
    print(f"  3. Press SPACE to confirm, ESC to skip")
    print(f"{'='*60}\n")
    
    tool = TopCalibrationTool()
    
    for image_path in image_paths:
        tool.calibrate_image(image_path)
    
    cv2.destroyAllWindows()
    
    tool.save_calibration()

if __name__ == "__main__":
    main()