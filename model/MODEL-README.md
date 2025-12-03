Front, top, and side models are seperated for now. they will eventually be integrated with a
view finder image distibutor to the different models.
They are tested individually for now. See notes at top of \*yolo_measurements.py files for individual details.

TO RUN: SIDE

cd model/side

python side_yolo_measurements.py \
 --model best.pt \
 --calibration side_calibration.json \
 --batch ../pictures/side/ \
 --debug

TO RUN: TOP

cd model/top

python top_yolo_measurements.py \
 --model best.pt \
 --calibration top_calibration.json \
 --batch ../pictures/top/ \
 --conf 0.3 \
 --debug

TODO FRONT

TRAIN

yolo segment train data=data.yaml model=yolov8n-seg.pt epochs=50 imgsz=640 batch=8
