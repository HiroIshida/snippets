Create a virtual CAN interface on Linux
```bash
sudo modprobe vcan
sudo ip link add dev vcan0 type vcan
sudo ip link set up vcan0
```

Run the motor emulator
```bash
bash start_motor.sh
```
and another terminal run the client
```bash
python3 linux_side.py
