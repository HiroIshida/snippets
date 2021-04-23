(load "package://pr2eus/pr2-interface.l")
(pr2-init)
;; *pr2* *ri*
(send *pr2* :angle-vector) ;; angle-vectorを見る
(send *pr2* :joint-list :name) ;; jointの名前一覧 (anglve-vector の順番どおりになってる)
(send *pr2* :methods :hogehoge) ;; *pr2* のインスタンスで使えるmethodの一覧のうちhogehogeという単語を含むものを表示. 使いたいメソッド名を忘れた場合に便利

;; *ri* (実機に繋いでいるときは実機, gazeboの場合はgazeboモデルのジョイントの状態を返す)
(send *ri* :state :potenito-vector)

;; 最初に*pr2*にアングルベクターをセットしてから, それを*ri*の方に送信する. 
;; angle-vector に引数をいれない場合は, get, 引数をつけた場合はsetの役割をはたす. 
(send *pr2* :angle-vector #f(40.5 -0.016206 0.756787 -0.185312 -16.0923 0.122556 -22.523 0.055081 0.000677 0.75984 0.104488 -16.1468 -0.079902 -22.4503 -0.047753 -0.013727 0.835757))
(send *pr2* :reset-manip-pose)
(send *ri* :angle-vector (send *pr2* :angle-vector) 3000) ;; 3[s]かけて目標のアングルに追従
 
;; 特定のアングルだけを変更したいときは次のようにする. 
(send *pr2* :torso_lift_joint :joint-angle 40)
(send *ri* :angle-vector (send *pr2* :angle-vector))

;; ikのターゲットになる座標系を作る
(setq *co-target* (make-coords :pos #f(500 -200 800) :rpy #f(0.0 0.1 0.1))) 
(objects (list *pr2* *co-target*)) ;; visualize してみる. 

(require "models/arrow-object.l") 
(setq *co-vis* (arrow))  
(send *co-vis* :newcoords (send *co-target* :copy-worldcoords))
(send *ri* :angle-vector (send *pr2* :angle-vector) 2000)

(objects (list *pr2* *co-vis*))

;; inverse kinematics を解く. 
(send *pr2* :rarm :inverse-kinematics *co-target* 
      :rotational-axis t
      :use-tosro t)
(send *ri* :angle-vector (send *pr2* :angle-vector)) ;; 送る



