(load "package://dxl_armed_turtlebot/euslisp/dxl-armed-turtlebot-interface.l")
(ros::roseus-add-msgs "jsk_recognition_msgs")

(dxl-armed-turtlebot-init)

(defvar *topic-name* "/camera/depth_registered/boxes")
(defvar *bounding-box-list* nil)
(setq *tfl* (instance ros::transform-listener :init))

(defun bounding-box-array-cb (msg)
  (unless (null msg) ;; set to global vars for debugging purpose
      (setq *frame-id* (concatenate string "/" (send msg :header :frame_id)))
      (setq *bounding-box-list* (send msg :boxes))
      (setq *box* (elt *bounding-box-list* 0))
      (setq *pose* (send *box* :pose))
      (setq *coords* (ros::tf-pose->coords *pose*))
      (setq *lt* (send *tfl* :lookup-transform "base_link" *frame-id* (ros::time 0))) ;; transform optical to base
      (setq *co-new* (send *lt* :transform *coords*)) ;; object coords
      )
  )

(ros::subscribe *topic-name* jsk_recognition_msgs::BoundingBoxArray #'bounding-box-array-cb 1)
(unix:sleep 2)
(ros::spin-once)

;(send *ri* :go-pos 0 0 360) 
(setq *mvbase-tgt* (send *co-new* :copy-worldcoords))
(send *mvbase-tgt* :translate #f(-200 0 0) :world)

(setq *c* (instance ros::simple-action-client :init "move_base" move_base_msgs::MoveBaseAction))
(setq goal (instance move_base_msgs::MoveBaseGoal :init))
(send goal :target_pose :header :stamp (ros::time-now))
(send goal :target_pose :header :frame_id "base_link") 

;(send goal :target_pose :pose :position (send (ros::coords->tf-pose *tgt-coords*) :position))
(send goal :target_pose :pose :position (send (ros::coords->tf-pose *mvbase-tgt*) :position))
(send goal :target_pose :pose :orientation :w 1) 

(send *c* :wait-for-server)
(send *c* :send-goal goal)
(ros::ros-info "wait for goal ...")
(send *c* :wait-for-result)
(ros::ros-info "done")
