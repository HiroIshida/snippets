import time
import numpy as np
import can
import struct
import random
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, default=0x01, help="CAN ID of the motor")
    args = parser.parse_args()
    can_id = args.id

    def build_feedback(err=1):
        pos = random.randint(0, 0x3FFF)      # 14bit
        vel = random.randint(0, 0x0FFF)      # 12bit
        torque = random.randint(0, 0x0FFF)   # 12bit
        t_mos = random.randint(20,40)
        t_rot = random.randint(20,40)

        d0 = (can_id & 0xFF) | ((err & 0x0F) << 4)
        d1 = (pos >> 8) & 0xFF
        d2 = pos & 0xFF
        d3 = (vel >> 4) & 0xFF
        d4 = ((vel & 0x0F) << 4) | ((torque >> 8) & 0x0F)
        d5 = torque & 0xFF
        return [d0, d1, d2, d3, d4, d5, t_mos, t_rot]


    bus = can.interface.Bus('vcan0', bustype='socketcan')
    bus.set_filters([{"can_id": can_id, "can_mask": 0x7FF}])  # all bits are relevant
    MASTER_ID = 0x00

    while True:
        msg = bus.recv()
        if msg is None:
            continue
        feedback = build_feedback()
        rsp = can.Message(arbitration_id=MASTER_ID,
                          data=feedback, is_extended_id=False)
        sleep_duration = np.random.uniform(0.0001, 0.0005)
        time.sleep(sleep_duration)  # simulate delay in motor response
        bus.send(rsp)
