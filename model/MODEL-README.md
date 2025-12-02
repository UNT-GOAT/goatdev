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
 --debug

TODO FRONT
