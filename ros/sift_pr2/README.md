## how to run
Inside PR2:
```bash
roslaunch detect_cans_in_fridge_201202 startup.launch
```
From my laptop: (run kalman filter)
```bash
rosun sift_pr2 estimator.py
```

From my laptop: (run realtime feedback)
```bsah
roseus feedback.l
```
## how to run bag file and testing kalman filter
```bash
roslaunch sift_pr2 pose_estimation.launch
```
