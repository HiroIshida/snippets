;; concatenate string
(concatenate string "hoge" "hoge")

;; faster angle-vector (nil -> rarm-controler maybe)
(send *ri* :angle-vector av :fast nil 0 :min-time 0.01)

;; make vector make-vector array initialize
(make-array '(3 3) :element-type :float)
(make-array 3 :element-type :float)


;; to specify maximum itr in IK; if :revert-if-fail is nil, then angle-vector will be overwritten even if ik-fail
(send* robot :inverse-kinematics-raw target-coords :rotation-axis (list t t) :move-target (list (send robot :larm :end-coords) (send robot :rarm :end-coords)) :avoid-collision-distance 5 :revert-if-fail nil :stop ik-stop-step :debug-view nil args)
