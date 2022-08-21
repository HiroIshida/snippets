(ros::load-ros-manifest "roseus")
(ros::roseus-add-srvs "euslisp_command_srvs")

(ros::roseus "euslisp-server")
(defun handle (req)
  (let* ((resp (send req :response))
         (command-string (send req :command))
         (command-expr (read-from-string command-string)))
    (eval command-expr)
    resp))
(ros::advertise-service "command" euslisp_command_srvs::EuslispDirectCommand #'handle)
(do-until-key
 (ros::spin-once))
