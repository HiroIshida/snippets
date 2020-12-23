;(setq libpython "/usr/lib/python2.7/config-x86_64-linux-gnu/libpython2.7.so")
(setq libpython "build/libmain.so")
(setq *pythonlib* (load-foreign libpython))

(defforeign pyimport-importmodule
            *pythonlib*
            "_PyImport_ImportModule"
            (:string) ;; module name
            :integer ;; module object pointer
            )

(defforeign pyobject_callmethod
            *pythonlib*
            "_PyObject_CallMethod"
            (:string :string) ;; module name
            :integer ;; module object pointer
            )

(setq *os* (pyimport-importmodule "os"))
(setq ret (pyobject_callmethod "os" "exit"))
(print ret)
 
