import numpy as np
import time
import argparse
import rospy
from datetime import datetime
from sensor_msgs.msg import Image
import matplotlib.pyplot as plt


class Accumulator:

    def __init__(self, topic1, topic2):
        self.seq_ts1 = []
        self.seq_ts2 = []
        self.sub1 = rospy.Subscriber(topic1, Image, self.callback1, queue_size=1)
        self.sub2 = rospy.Subscriber(topic2, Image, self.callback2, queue_size=1)

    def callback1(self, msg):
        self.seq_ts1.append(msg.header.stamp.to_sec())

    def callback2(self, msg):
        self.seq_ts2.append(msg.header.stamp.to_sec())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wide", action="store_true")
    args = parser.parse_args()

    rospy.init_node('topic_latency')
    if args.wide:
        topic1 = "/wide_stereo/left/image_raw"
        topic2 = "/wide_stereo/right/image_raw"
    else:
        topic1 = "/narrow_stereo/left/image_raw"
        topic2 = "/narrow_stereo/right/image_raw"
    acc = Accumulator(topic1, topic2)
    time.sleep(10.0)
    acc.sub1.unregister()
    acc.sub2.unregister()

    fig, ax = plt.subplots()
    print(f"ts1: {acc.seq_ts1}")
    print(f"ts2: {acc.seq_ts2}")
    diff = np.array(acc.seq_ts1) - np.array(acc.seq_ts2)
    print(diff)
    ax.plot(acc.seq_ts1, len(acc.seq_ts1)*[0], 'r.')
    ax.plot(acc.seq_ts2, len(acc.seq_ts2)*[0.1], 'b.')
    ax.set_ylim([-0.05, 0.15])
    plt.show()
