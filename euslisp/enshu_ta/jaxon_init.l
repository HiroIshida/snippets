(load "package://cart_humanoid/euslisp/cart_humanoid-interface.l")
(cart_humanoid-init)
(make-irtviewer)
(objects *robot*)

(send *ri* :angle-vector (send *robot* :reset-floating-pose))
(send *ri* :angle-vector (send *robot* :reset-pose))
