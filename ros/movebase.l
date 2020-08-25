(load "package://pr2eus/pr2-interface.l")
(ros::roseus "feedback-fridge" :anonymous t)
(pr2-init)
(send *ri* :move-trajectory-sequence 
      (list #f(0.3 0 0) #f(0.6 0 0) #f(0.9 0 0)) 
      (list 1000 2000 3000)
      :send-action t)
