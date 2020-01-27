(load "package://pr2eus/pr2-interface.l")
(ros::load-ros-manifest "roseus")
(ros::roseus "tracker" :anonymous t)
(pr2-init)

(setq *camera-frame* "/head_mount_kinect_rgb_optical_frame")
(setq *tfl* (instance ros::transform-listener :init))
(setq *msg* nil)
(setq *target-coords-base* nil) ;; wrt base_link
(setq *target-coords-camera* nil) ;; wrt base_link
(ros::subscribe "/object_pose" geometry_msgs::PoseStamped
                #'(lambda (msg) 
                    (setq *msg* msg)
                    (print "subscribed")
                    (let* ((pose (send msg :pose))
                           (lt (send *tfl* :lookup-transform
                                     "base_link" *camera-frame*
                                     (ros::time 0))))
                        (setq *target-coords-base* (send lt :transform (ros::tf-pose->coords pose)))
                        (setq *target-coords-camera* pose))))

(defun compute-pan-angle (co-base)
  (let* ((pos (send co-base :pos))
         (pan-angle (atan (aref pos 1) (aref pos 0)))
         (pan-angle-deg (/ (* pan-angle 180) 3.14)))
    pan-angle-deg))

(setq *node-state* 'init)

(defun handle-start (req)
  (let ((res (send req :response)))
    (speak-jp "ほげ")
    (setq *node-state* 'running)
    res))

(defun handle-terminate (req)
  (let ((res (send req :response)))
    (setq *node-state* 'terminated)
    res))

(ros::advertise-service "start_tracking" std_srvs::Empty #'handle-start)
(ros::advertise-service "terminate_tracking" std_srvs::Empty #'handle-terminate)

(defun tracking ()
  (unix:sleep 2)
  (ros::spin-once)
  (speak-en "start tracking")
  (loop 
    (unless (null *target-coords-base*)
      (ros::spin-once)
      (send *pr2* :angle-vector (send *ri* :state :potentio-vector))
      (setq angle-pan (compute-pan-angle *target-coords-base*))
      (print angle-pan)
      (send *pr2* :head_pan_joint :joint-angle angle-pan)
      (send *pr2* :head_tilt_joint :joint-angle 44.0)
      (send *ri* :angle-vector (send *pr2* :angle-vector) 2000)
      (unix:usleep 500000)
      )
    (when (eq *node-state* 'terminated) (return))
    ))

(unix:sleep 2)
(do-until-key
  (ros::spin-once)
  (print "hoge")
  (unless (eq *node-state* 'init) (return)))
(tracking)









