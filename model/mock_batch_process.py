"""
Batch Goat Processing Script

This is intended to mock the batch processing of goats through all angle models at one time, a POC.

Processes all goats from grouped_by_goat folder through side, top, and front models.
Combines measurements with weights and outputs complete goat data.

Directory structure expected:
grouped_by_goat/
├── weights.json              # {"1": 45.5, "2": 52.3, ...}
├── 1/
│   ├── side_1.jpg
│   ├── top_1.jpg
│   └── front_1.jpg
├── 2/
│   ├── side_2.jpg
│   ├── top_2.jpg
│   └── front_2.jpg
...

USAGE:
    cd model/

    python mock_batch_process.py \
        --path images/grouped_by_goat \
        --output batch_results.json \
        --debug

OUTPUT:
    batch_results.json - JSON file with all goat measurements, weights, and confidence scores.
    batch_debug/ - (optional) debug images from each model if --debug is used. Super helpful to see what the models saw.


TODO:
- Between top and front for width, if no head mask detected in front AND not theyre not within certain threshold of eachother, 
  use top width as fallback. Do not average. Theres a good change the head got measured into the width for front view.
"""

import sys
import json
from pathlib import Path
from datetime import datetime
import argparse

# Import measurement classes from each view
sys.path.append(str(Path(__file__).parent / "side"))
sys.path.append(str(Path(__file__).parent / "top"))
sys.path.append(str(Path(__file__).parent / "front"))

from side.side_yolo_measurements import YOLOGoatMeasurementsSide
from top.top_yolo_measurements import YOLOGoatMeasurementsTop
from front.front_yolo_measurements import YOLOGoatMeasurementsFront


class GoatProcessor:
    def __init__(self, goatpics_dir: Path, save_debug: bool = False):
        self.goatpics_dir = Path(goatpics_dir)
        self.save_debug = save_debug
        
        # Create debug folder structure if needed
        if self.save_debug:
            self.debug_dir = Path('batch_debug')
            self.debug_dir.mkdir(exist_ok=True)
        
        # Load weights
        weights_file = self.goatpics_dir / "weights.json"
        with open(weights_file, 'r') as f:
            self.weights = json.load(f)
        
        # Load calibrations
        with open('side/side_calibration.json', 'r') as f:
            side_cal = json.load(f)
            side_pixels_per_cm = side_cal['pixels_per_cm']
        
        with open('top/top_calibration.json', 'r') as f:
            top_cal = json.load(f)
            top_pixels_per_cm = top_cal['pixels_per_cm']
        
        with open('front/front_calibration.json', 'r') as f:
            front_cal = json.load(f)
            front_pixels_per_cm = front_cal['pixels_per_cm']
        
        # Initialize models
        print("Loading models...")
        self.side_model = YOLOGoatMeasurementsSide(
            'side/best.pt', 
            side_pixels_per_cm
        )
        self.top_model = YOLOGoatMeasurementsTop(
            'top/best.pt',
            top_pixels_per_cm
        )
        self.front_model = YOLOGoatMeasurementsFront(
            'front/best.pt',
            front_pixels_per_cm
        )
        
        # Enable debug mode if requested
        if self.save_debug:
            self.side_model.debug = True
            self.top_model.debug = True
            self.front_model.debug = True
        
        print("✓ Models loaded\n")
    
    def find_image(self, goat_folder: Path, view: str, goat_num: str):
        """Find image file for a view (handles .jpg, .JPG, etc.)"""
        pattern = f"{view}_{goat_num}.*"
        matches = list(goat_folder.glob(pattern))
        
        if not matches:
            return None
        return matches[0]
    
    def process_goat(self, goat_num: str):
        """Process one goat through all three views"""
        goat_folder = self.goatpics_dir / goat_num
        
        if not goat_folder.exists():
            return {
                'goat_id': goat_num,
                'success': False,
                'error': f'Folder not found: {goat_folder}'
            }
        
        # Find image files
        side_img = self.find_image(goat_folder, 'side', goat_num)
        top_img = self.find_image(goat_folder, 'top', goat_num)
        front_img = self.find_image(goat_folder, 'front', goat_num)
        print(f"  Top image: {top_img}")
        print(f"  Front image: {front_img}")
        
        # Check all images exist
        missing = []
        if not side_img: missing.append('side')
        if not top_img: missing.append('top')
        if not front_img: missing.append('front')
        
        if missing:
            return {
                'goat_id': goat_num,
                'success': False,
                'error': f'Missing images: {", ".join(missing)}'
            }
        
        # Get weight
        weight_lbs = self.weights.get(goat_num, None)
        
        # Process each view - models handle their own thresholds internally
        print(f"  Processing side view...")
        side_results = self.side_model.process_image(str(side_img))
        
        print(f"  Processing top view...")
        top_results = self.top_model.process_image(str(top_img))
        
        print(f"  Processing front view...")
        front_results = self.front_model.process_image(str(front_img))
        
        # Move debug images to goat-specific folder
        if self.save_debug:
            import shutil
            goat_debug_dir = self.debug_dir / goat_num
            goat_debug_dir.mkdir(exist_ok=True)
            
            # Copy debug images from each view's debug folder
            for view, img_path in [('side', side_img), ('top', top_img), ('front', front_img)]:
                debug_filename = f"debug_{img_path.name}"
                debug_path = Path(view) / 'debug' / debug_filename
                
                if debug_path.exists():
                    dest = goat_debug_dir / f"{view}_debug.jpg"
                    shutil.copy(debug_path, dest)
                else:
                    print(f"    Warning: Debug image not found at {debug_path}")
        
        # Extract key measurements
        measurements = {
            'head_height_cm': side_results.get('head_height_cm'),
            'withers_height_cm': side_results.get('withers_height_cm'),
            'rump_height_cm': side_results.get('rump_height_cm'),
            'top_body_width_cm': top_results.get('body_width_cm'),
            'front_body_width_cm': front_results.get('body_width_cm')
        }
        
        # Calculate average width
        widths = [
            measurements['top_body_width_cm'],
            measurements['front_body_width_cm']
        ]
        valid_widths = [w for w in widths if w is not None]
        measurements['avg_body_width_cm'] = (
            round(sum(valid_widths) / len(valid_widths), 2) 
            if valid_widths else None
        )
        
        # Combine results
        goat_data = {
            'goat_id': goat_num,
            'timestamp': datetime.now().isoformat(),
            'weight_lbs': weight_lbs,
            'measurements': measurements,
            'confidence_scores': {
                'side': side_results.get('yolo_confidence'),
                'top': top_results.get('yolo_confidence'),
                'front': front_results.get('yolo_confidence')
            },
            'all_views_successful': (
                side_results.get('success', False) and
                top_results.get('success', False) and
                front_results.get('success', False)
            ),
            'success': True
        }
        
        return goat_data
    
    def process_all(self, output_file: str):
        """Process all goats in GOATpics folder"""
        
        # Find all goat folders (numeric names)
        goat_folders = sorted([
            d.name for d in self.goatpics_dir.iterdir()
            if d.is_dir() and d.name.isdigit()
        ], key=int)
        
        print(f"Found {len(goat_folders)} goats to process\n")
        
        results = []
        successful_goats = 0
        
        for i, goat_num in enumerate(goat_folders, 1):
            print(f"[{i}/{len(goat_folders)}] Processing goat {goat_num}...")
            
            goat_data = self.process_goat(goat_num)
            results.append(goat_data)
            
            if goat_data.get('all_views_successful', False):
                successful_goats += 1
                weight = goat_data['weight_lbs']
                print(f"  ✓ Success! Weight: {weight}lbs\n")
            else:
                if not goat_data.get('success', False):
                    print(f"  ✗ Failed: {goat_data.get('error', 'Unknown error')}\n")
                else:
                    print(f"  ⚠ Partial success (some views failed)\n")
        
        # Save results
        output_data = {
            'processing_date': datetime.now().isoformat(),
            'total_goats': len(goat_folders),
            'successful_goats': successful_goats,
            'goats': results
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Processing complete!")
        print(f"{'='*60}")
        print(f"Total goats: {len(goat_folders)}")
        print(f"All views successful: {successful_goats}")
        print(f"Partial/failed: {len(goat_folders) - successful_goats}")
        print(f"\nResults saved to: {output_file}")
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description='Process all goats through side, top, and front models'
    )
    parser.add_argument(
        '--path',
        required=True,
        help='Path to grouped-by-goat folder'
    )
    parser.add_argument(
        '--output',
        default='batch_results.json',
        help='Output JSON file (default: batch_results.json)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Save debug images to batch_debug/ folder'
    )
    
    args = parser.parse_args()
    
    processor = GoatProcessor(args.path, save_debug=args.debug)
    processor.process_all(args.output)


if __name__ == "__main__":
    main()