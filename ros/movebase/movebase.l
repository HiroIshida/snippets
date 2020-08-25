(load "package://pr2eus/pr2-interface.l")
(ros::roseus "feedback-fridge" :anonymous t)
(pr2-init)
(send *ri* :move-trajectory-sequence 
      (list #f(0.5 0 -0.2) #f(1.0 0 -0.4) #f(1.5 0 -0.6) #f(2.0 0 -0.8))
      (list 500 1000 1500 2000)
      :send-action t)
