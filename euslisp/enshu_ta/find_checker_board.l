(load "package://dxl_armed_turtlebot/euslisp/dxl-armed-turtlebot-interface.l")
(load "models/chessboard-30-7x5-object.l")

(dxl-armed-turtlebot-init)
(setq *robot* *dxl-armed-turtlebot*)
(setq *target-object* (chessboard-30-7x5 :name "/test_object"))
(setq *object-detected* nil)
(setq *msg* nil)
(objects (list *target-object* *robot*))
(setq *tfb* (instance ros::transform-broadcaster :init))

(ros::roseus-add-msgs "jsk_recognition_msgs")
(ros::roseus-add-msgs "image_view2")
(ros::roseus-add-msgs "posedetection_msgs")

(defun objectdetection-cb (msg) ;; copied from enshu 
  (unless (null (send msg :objects))
      (setq *object-detected* t))
  (setq frame-id (concatenate string "/" (send msg :header :frame_id)))
  (mapcar #'(lambda (obj-pose)
              (let* (
                     (cam->obj-coords (ros::tf-pose->coords (send obj-pose :pose)))
                     (cam-coords (send (send *dxl-armed-turtlebot* :camera_rgb_optical_frame_lk) :copy-worldcoords)))
                (send *target-object* :newcoords (send cam-coords :transform cam->obj-coords))
                ))
          (send msg :objects))
  )
(ros::subscribe "/camera/rgb/ObjectDetection" posedetection_msgs::ObjectDetection #'objectdetection-cb)

(ros::advertise "/cmd_vel" geometry_msgs::Twist)
(defun rotate (angvel)
  (let ((msg (instance geometry_msgs::Twist :init)))
    (send msg :angular :z angvel)
    (ros::publish "/cmd_vel" msg)))

;(send *ri* :go-pos 0 0 -60)

(defun rotate-until-detected ()
  (setq *object-detected* nil)
    (loop 
        (rotate 0.5)
        (ros::spin-once)
        (when *object-detected* (return))))

(setq *pos* (send (send *target-object* :worldcoords) :pos))
(defun rotate-until-ceterized ()
  (loop 
    (let* ((pos (send (send *target-object* :worldcoords) :pos))
           (angle (atan2 (aref pos 1) (aref pos 0))))
      (print angle)
      (when (< (abs angle) 0.01) (return))
      (let ((Kp 1.0)) ;; p-gain 
          (rotate (* Kp angle)))
      (ros::spin-once))))

(rotate-until-detected)
(rotate-until-ceterized)


