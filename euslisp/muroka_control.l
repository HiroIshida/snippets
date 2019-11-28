(let ((dir "package://eus_qp/optmotiongen/euslisp/"))
  (dolist (file 
            (list "inverse-kinematics-wrapper.l" 
              "discrete-kinematics-configuration-task.l"
              "sqp-msc-optimization.l"))
    (load (concatenate string dir file))))

(require "irteus/demo/sample-arm-model.l")
(require "irteus/demo/sample-robot-model.l")
(require "models/arrow-object.l")

(defun sample-arm-reach-trajectory-ik
    (&key
     (pre-process-func)
     (post-process-func)
     (visualize-callback-func)
     &allow-other-keys
     )
  (setq *robot* (instance sarmclass :init))
  (setq *arrow-list* (mapcar #'(lambda (x) (instance arrow-object :init)) (make-list 10)))
  (send (elt *arrow-list* 0) :newcoords (make-coords :pos (float-vector 100 -400 300) :rpy (list (deg2rad 60) 0 0)))
  (dotimes (i (1- (length *arrow-list*)))
    (send (elt *arrow-list* (1+ i)) :newcoords
          (send (send (elt *arrow-list* i) :copy-worldcoords) :translate (float-vector 100 0 0))))
  (setq
   *ret*
   (send *robot*
         :inverse-kinematics-trajectory-optmotiongen
         (send-all *arrow-list* :copy-worldcoords)
         :move-target-list (send *robot* :end-coords)
         :debug-view (list :additional-draw-objects *arrow-list*)
         :rotation-axis-list (make-list (length *arrow-list*) :initial-element nil)
         :stop 50
         :min-loop 30
         :adjacent-regular-scale 1e-5
         :pre-process-func pre-process-func
         :post-process-func post-process-func
         ))
  (objects (append (list *robot*) *arrow-list*))
  (visualize-trajectory-result *ret* :visualize-callback-func visualize-callback-func)
  *ret*
  )

(sample-arm-reach-trajectory-ik)
