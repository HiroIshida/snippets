(defun hoge (x)
  (catch 'unko
    (huga x)
    (print "dicky")))

(defun huga (x)
  (if (> x 2)
  (throw 'unko "fuck")))

