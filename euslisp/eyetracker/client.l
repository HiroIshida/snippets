(ros::load-ros-manifest "roseus")
(ros::roseus "tester" :anonymous t)

(defun start-tracking () 
  (let ((service-name "start_tracking"))
      (ros::wait-for-service service-name)
      (ros::service-call service-name (instance std_srvs::EmptyRequest :init))))

(defun terminate-tracking () 
  (let ((service-name "terminate_tracking"))
      (ros::wait-for-service service-name)
      (ros::service-call service-name (instance std_srvs::EmptyRequest :init))))
