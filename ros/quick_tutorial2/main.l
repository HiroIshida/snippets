(load "package://pr2eus/pr2-interface.l") ;; load library to use pr2 stuff
(require "models/arrow-object.l") ;; load library to visualize coordinate
(ros::roseus-add-msgs "jsk_recognition_msgs")  ;; to use jsk_recognition_msgs::BoundingBoxArray
(ros::roseus "main" :anonymous t)  ;; define node

;; define callback
(setq *box* nil)  ;; defeine global variable to store subscribed message
(defun callback (msg)
  (print "subscribed")
  (let ((boxes (send msg :boxes)))
    (if (> (length boxes) 1) ;; set *box* if detected num of boxes is 1
      (return-from callback nil)
      (setq *box* (elt boxes 0))
      )))

;; instantiate subscriber
(ros::subscribe
  "/boxes"  ;; topic name
  jsk_recognition_msgs::BoundingBoxArray  ;; topic type
  #'callback)  ;; callback

;; instantiate *pr2* and *ri*. Both *pr2* and *ri* are global varaibles.
(pr2-init)

;; set local pr2 model configuration
(send *pr2* :reset-manip-pose)
(send (send *pr2* :torso_lift_joint) :joint-angle 300)
(send (send *pr2* :r_shoulder_lift_joint) :joint-angle -30)

;; reflect the local pr2 model configuration to real one
(send *ri* :angle-vector (send *pr2* :angle-vector))
(send *ri* :wait-interpolation)  ;; wait until reflection complete

;; call callback until *box* is set
(while (null *box*)
    (print *box*)
    (unix::sleep 1)
    (ros::spin-once))
(print "box is set")

;; extract position of box and create target-coords
(setq *target-coords* nil)
(let* ((pos-msg (send (send *box* :pose) :position))
       (x-m (send pos-msg :x))
       (y-m (send pos-msg :y))
       (z-m (send pos-msg :z))
       (x-mm (* x-m 1000))
       (y-mm (* y-m 1000))
       (z-mm (* z-m 1000)))
  (setq *target-coords* (make-coords :pos (float-vector x-mm y-mm z-mm)))
  (send *target-coords* :rotate 0.7 :z :local)
  (send *target-coords* :rotate 0.4 :y :local))

(setq *pre-target-coords* (send *target-coords* :copy-worldcoords))
(send *pre-target-coords* :translate #f(-80 0 0) :local)

;; visualize geometric relation between pr2 and target-coords
(setq *vis-taget-coords* (arrow))
(send *vis-taget-coords* :newcoords (send *target-coords* :copy-worldcoords))
(setq *vis-pre-target-coords* (arrow))
(send *vis-pre-target-coords* :newcoords (send *pre-target-coords* :copy-worldcoords))
(objects (list *pr2* *vis-taget-coords* *vis-pre-target-coords*))

;; reach to *pre-target-coords*
(send *pr2* :rarm :inverse-kinematics
      *pre-target-coords*
      :rotation-axis t 
      :check-collision nil
      :use-torso nil)
(send *ri* :angle-vector (send *pr2* :angle-vector))
(send *ri* :wait-interpolation)

;; reach to *target-coords*
(send *pr2* :rarm :inverse-kinematics
      *target-coords*
      :rotation-axis t 
      :check-collision nil
      :use-torso nil)
(send *ri* :angle-vector (send *pr2* :angle-vector))
(send *ri* :wait-interpolation)


;; translate rarm a little bit farther
(send *pr2* :rarm :move-end-pos #f(50 0 0) :local)
(send *ri* :angle-vector (send *pr2* :angle-vector))
(send *ri* :wait-interpolation)

;; grasp!
(send *ri* :start-grasp :rarm)
