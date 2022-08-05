(load "package://pr2eus/pr2-interface.l")
(ros::roseus "feedback-fridge" :anonymous t)

(setq *tfl* (instance ros::transform-listener :init))
(setq *target-co* nil)
(ros::subscribe "fridge_pose" geometry_msgs::PoseStamped
                #'(lambda (msg) 
                    (print "subscribed")
                    (let ((pose (send msg :pose))
                          (lt (send *tfl* :lookup-transform "base_link" "map" (ros::time 0))))
                      (setq *target-co* (send lt :transform (ros::tf-pose->coords pose))))))

(unix:sleep 2)
(ros::spin-once)
