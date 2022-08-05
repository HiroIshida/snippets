(load "package://pr2eus/pr2-interface.l")
(pr2)
(send *pr2* :reset-pose)
(setq *ec* (send *pr2* :rarm :end-coords :parent))
(setq *l* (send *pr2* :link-list *ec*))
(setq link-r-wrist (elt *l* 7))

(setq *co-center* (make-cascoords))
(send *co-center* :newcoords (send link-r-wrist :copy-worldcoords))
(send *co-center* :translate #f(180 0 250) :local)
(send link-r-wrist :assoc *co-center*)

(send *pr2* :init-pose)

(defun compute-jacobian ()
  (let* ((link (cdr (send *pr2* :link-list (send (send *pr2* :rarm :end-coords) :parent))))
         (j (send *pr2* :calc-jacobian-from-link-list link
                  :move-target *co-center*
                  :rotation-axis nil
                  :translation-axis t))
         (j# (sr-inverse j)))
    j#))

(setq j (compute-jacobian))
(setq vec-surface (transform (transpose (send *co-center* :worldrot)) #f(0 1 0)))
(transform j# vec-surface)
;(print (transform (transpose (send *co-center* :worldrot)) #f(0 1 0)))


(objects (list *pr2* *co-center*))
(print (send *co-center* :worldrot))

;(setq j (compute-jacobian))
 
#|
(send *pr2* :rarm :inverse-kinematics
      (make-coords :pos #f(1000 0 700) :rpy #f(0 0 0))
      :move-target *co-center*
      :rotation-axis nil :use-torso nil)
|#
