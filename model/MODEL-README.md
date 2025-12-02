Front, top, and side models are seperated for now. they will eventually be integrated with a
view finder image distibutor to the different models.
They are tested individually for now. See notes at top of \*yolo_measurements.py files for individual details.

TO RUN: SIDE

cd model/side

python side_yolo_measurements.py \
 --model YOLO_MODEL_STUFF/runs/segment/train/weights/best.pt \
 --calibration calibration.json \
 --batch side_pics/ \
 --debug

TODO FRONT AND TOP
