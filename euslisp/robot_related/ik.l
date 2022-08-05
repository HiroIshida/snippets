;; set maximum angle which will be considered in solving ik
(load "package://pr2eus/pr2-interface.l")
(let ((dir "package://eus_qp/optmotiongen/euslisp/"))
  (dolist (file 
            (list "inverse-kinematics-wrapper.l" 
              "discrete-kinematics-configuration-task.l"
              "sqp-msc-optimization.l"))
    (load (concatenate string dir file))))

(pr2)
(objects *pr2*)
(setq *cube* (make-cube 100 100 100))
(send *cube* :translate #f(500 -300 500))
(send *pr2* :inverse-kinematics-optmotiongen
      (make-coords :pos #f(800 -200 500))
      ;:link-list (cddr (send *pr2* :link-list (send (send *pr2* :rarm :end-coords) :parent)))
      :link-list (send *pr2* :link-list (send (send *pr2* :rarm :end-coords) :parent))
      :move-target (send *pr2* :rarm :end-coords)
      :translation-axis t
      :rotation-axis nil
      :debug-view t
      :obstacles (list *cube*)
      ;:posture-joint-list (list (send *pr2* :rarm :shoulder-r))
      ;:posture-joint-angle-list (list 0)
      ;:target-posture-scale 1e-3
      )

(objects (list *cube* *pr2*))

