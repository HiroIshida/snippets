# restrict core to 0,1
taskset -c "0,1" python3 compare.py
taskset -c "0,1" python3 compare.py --use_gpu
taskset -c "0,1" python3 compare.py --use_onnxruntime

# no restriction
python3 compare.py
python3 compare.py --use_gpu
python3 compare.py --use_onnxruntime
