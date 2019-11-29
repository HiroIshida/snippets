(load "package://pr2eus/pr2-interface.l")
(pr2)

(defun compute-jacobian (robot arm)
  (let* ((link (cddr (send robot :link-list (send (send robot arm :end-coords) :parent))))
         (j (send robot :calc-jacobian-from-link-list link
                  :move-target (send robot arm :end-coords)
                  :rotation-axis t
                  :translation-axis nil))
         (j# (sr-inverse j)))
    j#))



