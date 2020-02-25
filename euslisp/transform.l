;; for example tf 
;; tf_relative = listener.lookupTransform('/handle', '/r_gripper_tool_frame', rospy.Time(0))
(defun make-quaternion-msg (q-lst)
  (let ((key-lst '(:x :y :z :w))
        (idx 0)
        (q-msg (instance geometry_msgs::Quaternion :init)))
    (dolist (key key-lst)
      (send q-msg key (elt q-lst idx))
      (incf idx))
    q-msg))

(defun tf->coords (tf)
  (let ((co (make-coords))
        q-msg
        rot)
        
    (setq q-msg (make-quaternion-msg (second tf)))
    (setq rot (ros::tf-quaternion->rot q-msg))

    (send co :translate (scale 1000 (apply #'float-vector (first  tf))) :world)
    (send co :rotate-with-matrix rot :world)
    co))
