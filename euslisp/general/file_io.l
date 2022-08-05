(defun write-to-tmp nil
    (with-open-file (out-stream "tmp" :direction :output)
      (dotimes (i 10)
        (print i out-stream))))

(defun read-from-tmp nil
    (with-open-file (inp-stream "tmp" :direction :input)
      (let ((lst nil))
          (loop
              (let ((line (read-line inp-stream nil)))
                (unless line (return))
                (push line lst)
                ))
          (nreverse lst))))

(write-to-tmp)
(setq lst (read-from-tmp))

