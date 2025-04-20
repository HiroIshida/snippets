trap 'echo "Killing all motors..."; kill 0; exit' SIGINT
for i in $(seq 1 7); do
    echo "Starting motor_side.py with ID 0x0$i"
    python3 motor_side.py --id $i &
done
wait
