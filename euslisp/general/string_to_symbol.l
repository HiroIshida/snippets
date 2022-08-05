;; use read-from-string
(load "package://pr2eus/pr2-interface.l")
(pr2)

(setq method-name ":angle-vector")
(print (send *pr2* (read-from-string method-name)))
;; #f(50.0 60.0 74.0 70.0 -120.0 20.0 -30.0 180.0 -60.0 74.0 -70.0 -120.0 -20.0 -30.0 180.0 0.0 0.0)
