#!/usr/bin/env roseus
(load "package://cart_humanoid/euslisp/cart_humanoid-interface.l")
(ros::roseus-add-msgs "speech_recognition_msgs")

(ros::roseus "cart_humanoid_ik_node")
(cart_humanoid-init)

(setq *msg* nil)

(defun cb (msg)
  (setq *msg* msg))

(ros::subscribe "/speech_to_text" speech_recognition_msgs::SpeechRecognitionCandidates #'cb)

(ros::rate 10)
(while (ros::ok)
       (unless (null *msg*)
           (setq *likely-word* (elt (send *msg* :transcript) 0))
           (print *likely-word*)
           (when (string-equal *likely-word* "きつね")
             (send *ri* :go-velocity 1.0 0 0))
           (when (string-equal *likely-word* "わかめ")
             (send *ri* :go-velocity -1.0 0 0))
           (setq *msg* nil)
           )
       (ros::spin-once)
       (ros::sleep))
