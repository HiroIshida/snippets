import time
import can
import struct
import random
import time

bus = can.interface.Bus(channel="vcan0", bustype="socketcan", receive_own_messages=False)
control_ids = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07]

def build_mit_frame():
    p_des = random.randint(0, 0x3FFF)
    v_des = random.randint(0, 0x0FFF)
    Kp    = random.randint(0, 0x01F4)
    Kd    = random.randint(0, 0x0005)
    t_ff  = random.randint(0, 0x0FFF)

    b0 = (p_des >> 8) & 0xFF
    b1 = p_des & 0xFF
    b2 = (v_des >> 4) & 0xFF
    b3 = ((v_des & 0x0F) << 4) | ((Kp >> 8) & 0x0F)
    b4 = Kp & 0xFF
    b5 = (Kd >> 4) & 0xFF
    b6 = ((Kd & 0x0F) << 4) | ((t_ff >> 8) & 0x0F)
    b7 = t_ff & 0xFF
    return [b0, b1, b2, b3, b4, b5, b6, b7]


def decode_feedback(data):
    d0 = data[0]
    can_id =  d0 & 0x0F
    err    = (d0 >> 4) & 0x0F
    pos = ((data[1] << 8) | data[2]) & 0x3FFF
    vel = ((data[3] << 4) | (data[4] >> 4)) & 0x0FFF
    torque = (((data[4] & 0x0F) << 8) | data[5]) & 0x0FFF
    t_mos = data[6]
    t_rot = data[7]
    return can_id, err, pos, vel, torque, t_mos, t_rot


def main_brocking():
    duration = 0.001
    while True:
        ts = time.time()
        data = build_mit_frame()
        for cid in control_ids:
            msg = can.Message(arbitration_id=cid,
                              data=bytearray(data),
                              is_extended_id=False)
            try:
                bus.send(msg)
            except can.CanError:
                print("fail")
            msg = bus.recv(timeout=0.1)
        elapsed = time.time() - ts
        hz = 1 / elapsed
        print(f"hz: {hz:.2f} Hz")
        remain = duration - elapsed
        if remain < 0:
            print("Warning: send time exceeded duration")
        else:
            time.sleep(remain)

def main_nonblocking():
    duration = 0.001
    while True:
        ts = time.time()
        data = build_mit_frame()
        # batch send
        for cid in control_ids:
            msg = can.Message(arbitration_id=cid,
                              data=bytearray(data),
                              is_extended_id=False)
            try:
                bus.send(msg)
            except can.CanError:
                print("fail")

        # batch receive
        for cid in control_ids:
            msg = bus.recv(timeout=0.1)
            if msg is not None:
                can_id, err, pos, vel, torque, t_mos, t_rot = decode_feedback(msg.data)
                print(f"ID: {can_id}, err: {err}, pos: {pos}, vel: {vel}, torque: {torque}, t_mos: {t_mos}, t_rot: {t_rot}")
            else:
                print("No message received")

        elapsed = time.time() - ts
        hz = 1 / elapsed
        print(f"hz: {hz:.2f} Hz")
        remain = duration - elapsed
        if remain < 0:
            print("Warning: send time exceeded duration")
        else:
            time.sleep(remain)


if __name__ == "__main__":
    # main_brocking()
    main_nonblocking()
