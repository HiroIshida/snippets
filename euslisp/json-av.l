(load "package://roseus_mongo/euslisp/json/json-encode.l")
(load "package://roseus_mongo/euslisp/json/json-decode.l")

;; consider we have 4 avs
(setq *av1* #f(1 0 0 0)) 
(setq *av2* #f(0 1 0 0)) 
(setq *av3* #f(0 0 1 0)) 
(setq *av4* #f(0 0 0 1)) 

;; because to use json-encode, data of the alist cannot be a list, so we must convert it to a list 
;; by using the following function for example 
(defun vec-to-list (vec)
  (let ((lst-ret nil))
    (dotimes (i (length vec))
      (push (aref vec i) lst-ret))
    (reverse lst-ret)))

(setq *av-list* 
      (list 
        (vec-to-list *av1*)
        (vec-to-list *av2*)
        (vec-to-list *av3*)
        (vec-to-list *av4*)))

;; Note that even if you have only one pair of key and data (in this sample :av-data and *av-list*) you must wrap it by a list
(setq *alist* (list (cons :av-data *av-list*)))

;; save it
(defun save-jsonstring (filename jsonstring)
  (let ((out (open filename :direction :output)))
    (json::encode-element jsonstring out)
    (close out)))
(save-jsonstring "data-av.json" *alist*)

;; load the file
(defun load-jsonfile (filename)
  (let* ((in (open filename :direction :input))
         (alist (json::parse-object in)))
    (close in)
    alist))
(setq *alist-load* (load-jsonfile "data-av.json"))
(setq *av-list-load* (cdr (assoc :av-data *alist-load*) ))

;; now you have list of list in *av-list-load* 
;; you must convert it to float-vector again
;; converting a single list to a vector can be done by the following funciton for example
(defun lst->vec (lst)
  (apply #'float-vector lst))

;; you can map lst->vec function for each element of *av-list-load* by the following 
;; and now you have list of vectors
(setq *list-of-vector* (mapcar #'lst->vec *av-list-load*))

;; finally, mapcar and apply are so useful in lisp, so I recommend you to check 
;; http://jtra.cz/stuff/lisp/sclr/mapcar.html
;; http://jtra.cz/stuff/lisp/sclr/apply.html














