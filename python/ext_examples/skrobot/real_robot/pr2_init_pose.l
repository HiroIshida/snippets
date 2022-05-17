(load "package://pr2eus/pr2-interface.l")
(pr2-init)
(send *pr2* :init-pose)
(send *ri* :angle-vector (send *pr2* :angle-vector))
(send *ri* :wait-interpolation)

(send (send *pr2* :r_wrist_roll_joint) :joint-angle (* 180 -4))
(send *ri* :angle-vector (send *pr2* :angle-vector)
