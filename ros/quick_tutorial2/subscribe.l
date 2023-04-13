(ros::roseus-add-msgs "jsk_recognition_msgs")  ;; to use box array msg

(ros::roseus "main" :anonymous t)
(setq *box* nil)  ;; defeine global variable to store subscribed message

(defun callback (msg)
  (print "subscribed")
  (let ((boxes (send msg :boxes)))
    (if (< (length boxes) 2)
      (return-from callback nil)
      (setq *box* (elt boxces 0))
      )))

(ros::subscribe "/boxes" jsk_recognition_msgs::BoundingBoxArray #'callback)
(do-until-key
  (ros::spin-once))
