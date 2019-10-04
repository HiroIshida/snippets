(load "package://roseus_mongo/euslisp/json/json-encode.l")
(load "package://roseus_mongo/euslisp/json/json-decode.l")

;; make an association list 
;; see -> http://www.cs.cmu.edu/Groups/AI/html/cltl/clm/node153.html
(setq *pair1* `(:height . 163))
(setq *pair2* `(:weight . 60))
(setq *pair3* `(:age . 99))
(setq *alist* (list (cons :height 163) (cons :weight 60) (cons :age 99)))

;; save jsonstring:
(defun save-jsonstring (filename jsonstring)
  (let ((out (open filename :direction :output)))
    (json::encode-element jsonstring out)
    (close out)))
(save-jsonstring "data.json" *alist*)

;; load json
(defun load-jsonfile (filename)
  (let* ((in (open filename :direction :input))
         (alist (json::parse-object in)))
    (close in)
    alist))
(setq *alist-load* (load-jsonfile "data.json"))

;; you can extract the data
(assoc :height *alist-load*)

