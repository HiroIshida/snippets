(load "package://fetcheus/fetch-interface.l") 
(load "package://roseus_mongo/euslisp/json/json-encode.l")
(load "package://roseus_mongo/euslisp/json/json-decode.l")

(fetch)
(send *fetch* :init-pose)
(setq *links* (send *fetch* :links))
(setq *linknames* (mapcar #'(lambda (lk) (send lk :name)) *links*))
(setq *joints* (send *fetch* :joint-list))
(setq *jointnames* (mapcar #'(lambda (jt) (send jt :name)) *joints*))
(setq *limit-lower* (mapcar #'(lambda (jt) (send jt :min-angle)) *joints*))
(setq *limit-upper* (mapcar #'(lambda (jt) (send jt :max-angle)) *joints*))
(setq *limits* (mapcar #'cons *limit-lower* *limit-upper*))

(defun random-angle-vector ()
  (let (av-rand-tmp)
    (dolist (lm *limits*) 
      (let* ((low (car lm)) (up (cdr lm)) (diff (- up low)))
        (when (or (= low *inf*) (= up *inf*))
          (setq low -100.0)
          (setq up 100.0)
          (setq diff 200.0))
        (let ((rn (+ (random (* 1.0 diff)) low)))
          (push rn av-rand-tmp))))
    (apply #'float-vector (reverse av-rand-tmp))))

(defun random-angle-vector-list ()
  (let ((random-avs))
    (dotimes (i 20)
      (push (random-angle-vector) random-avs))
    random-avs))

(defun get-linkpoints (av)
  (let ((robot (fetch)));; workaround 
    (send robot :angle-vector av)
    (mapcar  #'(lambda (lk) (send lk :worldpos)) (send robot :links))))

(defun save-jsonstring (filename jsonstring)
  (let ((out (open filename :direction :output)))
    (json::encode-element jsonstring out)
    (close out)))

(defun vec-to-list (vec)
  (let ((lst-ret nil))
    (dotimes (i (length vec))
      (push (aref vec i) lst-ret))
    (reverse lst-ret)))

(defun vec-list-to-list-list (vec-list)
  (mapcar #'vec-to-list vec-list))

(setq *random-avs* (random-angle-vector-list))
(setq *linkpoints-list* (mapcar #'get-linkpoints *random-avs*))

;; because roseus json cannot encode float vector we must convert them to list of list
(setq *random-angle-list-list* (mapcar #'vec-to-list *random-avs*))
(setq *linkpoint-list-list* (mapcar #'vec-list-to-list-list *linkpoints-list*))

(setq *data-tobe-saved*
      (list
        (cons :linklist *linknames*)
        (cons :jointlist *jointnames*)
        (cons :avlist *random-angle-list-list*)
        (cons :pointslist *linkpoint-list-list*)))
(save-jsonstring "testdata.json" *data-tobe-saved*)
