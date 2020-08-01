(load "package://pr2eus/pr2-interface.l")
(ros::roseus "feedback-fridge" :anonymous t)

(setq *tfl* (instance ros::transform-listener :init))
(setq *target-co* nil)
(ros::subscribe "fridge_pose" geometry_msgs::PoseStamped
                #'(lambda (msg) 
                    (print "subscribed")
                    (let ((pose (send msg :pose))
                          (lt (send *tfl* :lookup-transform "base_link" "map" (ros::time 0))))
                      (setq *target-co* (send lt :transform (ros::tf-pose->coords pose)))
                      (send *target-co* :translate #f(-100 0 0) :local))))

(pr2-init)
(ros::spin-once)

;; initialize 
(send *pr2* :init-pose)
(send *pr2* :rarm :inverse-kinematics
      (make-coords :pos #f(200 -500 1300) :rot #f(0 0 0))
      :rotation-axis nil :use-torso nil)

(send *pr2* :larm :inverse-kinematics
      (make-coords :pos #f(200 500 1300) :rot #f(0 0 0))
      :rotation-axis nil :use-torso nil)
(objects *pr2*)
(send *ri* :angle-vector (send *pr2* :angle-vector) 3000)
(send *ri* :wait-interpolation)

(ros::rate (/ 1.0 0.1))
(dotimes (i 10000)
  (setq *av-pre* (send *pr2* :angle-vector))
  (bench (send *pr2* :larm :inverse-kinematics
        *target-co*
        :revert-if-fail t
        :loop 20
        :rotation-axis t :use-torso nil))
  (setq *av-post* (send *pr2* :angle-vector))
  (setq *av-diff* (v- *av-post* *av-pre*))
  (setq *av-com*  (v+ *av-pre* (scale 2.5 *av-diff*)))
  (send *pr2* :angle-vector (send *ri* :state :potentio-vector))
  (send *ri* :angle-vector *av-post* 500 :larm-controller 0 :min-time 0.3 :scale 5.0)
  (ros::sleep)
  (ros::spin-once)
  )

