
"""
NOT FOR FREQUENT USE - Find 'Measurements Calibration' in side_yolo_measurements.py for more details.
"""

import cv2
import numpy as np
import json
from pathlib import Path

class CalibrationTool:
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Scale image to fit screen if needed
        max_height = 900
        height, width = self.image.shape[:2]
        if height > max_height:
            scale = max_height / height
            new_width = int(width * scale)
            self.display_image = cv2.resize(self.image, (new_width, max_height))
            self.scale = scale
        else:
            self.display_image = self.image.copy()
            self.scale = 1.0
        
        self.points = []
        self.cm_values = [0, 10, 20, 30, 40, 50, 60, 70]
        self.window_name = "Calibration Tool - Multi-point"
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < len(self.cm_values):
                # Convert back to original image coordinates
                orig_x = int(x / self.scale)
                orig_y = int(y / self.scale)
                self.points.append((orig_x, orig_y))
                
                # Draw on display image (scaled coordinates)
                cv2.circle(self.display_image, (x, y), 5, (0, 255, 0), -1)
                
                cm_value = self.cm_values[len(self.points) - 1]
                cv2.putText(self.display_image, f"{cm_value}cm", (x + 10, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                print(f"✓ {cm_value}cm mark: ({orig_x}, {orig_y})")
                
                # Draw lines between consecutive points
                if len(self.points) > 1:
                    pt1 = (int(self.points[-2][0] * self.scale), int(self.points[-2][1] * self.scale))
                    pt2 = (x, y)
                    cv2.line(self.display_image, pt1, pt2, (0, 255, 0), 2)
                
                # If we have all points, calculate calibration
                if len(self.points) == len(self.cm_values):
                    self.calculate_calibration()
                else:
                    next_cm = self.cm_values[len(self.points)]
                    print(f"Now click on the {next_cm}cm mark...")
                
                cv2.imshow(self.window_name, self.display_image)
    
    def calculate_calibration(self):
        """Calculate pixels per inch from all points using averaging"""
        
        # Calculate spacing between each consecutive pair of points
        spacings = []
        for i in range(len(self.points) - 1):
            p1 = self.points[i]
            p2 = self.points[i + 1]
            
            # Distance in pixels
            pixel_distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            
            # This represents 10cm
            pixels_per_10cm = pixel_distance
            spacings.append(pixels_per_10cm)
            
            cm_start = self.cm_values[i]
            cm_end = self.cm_values[i + 1]
            print(f"  {cm_start}cm → {cm_end}cm: {pixel_distance:.2f} pixels")
        
        # Calculate average spacing for 10cm
        avg_pixels_per_10cm = np.mean(spacings)
        std_pixels_per_10cm = np.std(spacings)
        
        # Calculate pixels per cm
        pixels_per_cm = avg_pixels_per_10cm / 10.0
        
        # Convert to pixels per inch (1 inch = 2.54 cm)
        pixels_per_inch = pixels_per_cm * 2.54
        
        print(f"\n{'='*60}")
        print(f"CALIBRATION RESULTS")
        print(f"{'='*60}")
        print(f"Average spacing per 10cm: {avg_pixels_per_10cm:.2f} pixels")
        print(f"Spacing std deviation: {std_pixels_per_10cm:.2f} pixels")
        print(f"Spacing variation: {(std_pixels_per_10cm/avg_pixels_per_10cm)*100:.1f}%")
        print(f"\nPixels per cm: {pixels_per_cm:.2f}")
        print(f"Pixels per inch: {pixels_per_inch:.2f}")
        
        # Warn if variation is too high
        if std_pixels_per_10cm / avg_pixels_per_10cm > 0.05:  # More than 5% variation
            print(f"\n⚠️  WARNING: High variation in spacing ({(std_pixels_per_10cm/avg_pixels_per_10cm)*100:.1f}%)")
            print("   Consider recalibrating for better accuracy")
        
        print(f"\nPress 's' to save calibration, 'r' to reset, 'q' to quit")
        
        self.pixels_per_cm = pixels_per_cm
        self.pixels_per_inch = pixels_per_inch
        self.spacings = spacings
        self.avg_spacing = avg_pixels_per_10cm
        self.std_spacing = std_pixels_per_10cm
    
    def save_calibration(self, output_file: str = "calibration.json"):
        """Save calibration to file"""
        calibration_data = {
            'image_path': str(self.image_path),
            'calibration_points': {
                f'{cm}cm': list(point) 
                for cm, point in zip(self.cm_values, self.points)
            },
            'num_points': len(self.points),
            'spacings_per_10cm': [float(s) for s in self.spacings],
            'average_spacing_10cm': float(self.avg_spacing),
            'std_deviation_10cm': float(self.std_spacing),
            'variation_percent': float((self.std_spacing / self.avg_spacing) * 100),
            'pixels_per_cm': float(self.pixels_per_cm),
            'pixels_per_inch': float(self.pixels_per_inch),
            'image_width': self.image.shape[1],
            'image_height': self.image.shape[0]
        }
        
        with open(output_file, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        print(f"\n✓ Calibration saved to {output_file}")
        print(f"\nYou can now run:")
        print(f"  python yolo_measurements.py --batch usable/side/ --calibration {output_file}")
    
    def reset(self):
        """Reset points and start over"""
        self.points = []
        self.display_image = self.image.copy()
        if self.scale != 1.0:
            max_height = 900
            height, width = self.image.shape[:2]
            new_width = int(width * self.scale)
            self.display_image = cv2.resize(self.image, (new_width, max_height))
        print("\nReset! Click on the 0cm mark...")
        cv2.imshow(self.window_name, self.display_image)
    
    def run(self):
        """Run the calibration tool"""
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # Add instructions to image
        instructions = self.display_image.copy()
        cv2.putText(instructions, "Click on each cm mark: 0, 10, 20, 30, 40, 50, 60, 70", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(instructions, "Start at bottom (0cm), work your way up", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(instructions, "Press 's' to save, 'r' to reset, 'q' to quit", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        self.display_image = instructions
        cv2.imshow(self.window_name, self.display_image)
        
        print("Calibration Tool - Multi-point (0-70cm)")
        print("="*60)
        print("Instructions:")
        print("1. Click on each mark in order: 0cm, 10cm, 20cm, 30cm, 40cm, 50cm, 60cm, 70cm")
        print("2. Start at the bottom (0cm) and work your way up")
        print("3. The tool will average all spacings for accurate calibration")
        print("4. Press 's' to save calibration")
        print("5. Press 'r' to reset and try again")
        print("6. Press 'q' to quit")
        print("="*60)
        print("\nClick on the 0cm mark...")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                if len(self.points) == len(self.cm_values):
                    self.save_calibration()
                    break
                else:
                    print(f"Please select all {len(self.cm_values)} points first! ({len(self.points)}/{len(self.cm_values)} done)")
            elif key == ord('r'):
                self.reset()
        
        cv2.destroyAllWindows()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Manual Calibration Tool - Multi-point Centimeters')
    parser.add_argument('image', help='Path to side-view image with visible cm ruler (0-70cm)')
    parser.add_argument('--output', default='calibration.json',
                       help='Output calibration file')
    
    args = parser.parse_args()
    
    try:
        tool = CalibrationTool(args.image)
        tool.run()
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())