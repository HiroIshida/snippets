import argparse
import time
import rospy
from rostopic import ROSTopicHz, ROSTopicDelay, ROSTopicBandwidth, get_topic_class
from typing import Literal, List, Dict, Any, Optional
from collections import defaultdict


class RosparamMonitor:
    param_table: Dict[str, Any]
    prefix: Optional[str]

    def __init__(self, prefix: Optional[str]):
        self.prefix = prefix
        if prefix is None:
            self.param_table = {}
        else:
            all_params = rospy.get_param_names()
            filtered_params = [param for param in all_params if param.startswith(prefix)]
            self.param_table = {param: rospy.get_param(param) for param in filtered_params}

    def detect_change(self, event) -> Dict[str, Any]:
        change_dict = {}
        for param, value in self.param_table.items():
            new_value = rospy.get_param(param)
            if new_value != value:
                self.param_table[param] = new_value
                param_without_prefix = param[len(self.prefix):]
                change_dict[param_without_prefix] = (value, new_value)
        return change_dict


class TopicMonitor:

    def __init__(self, topic_name: str, monitor_params_prefix: Optional[str] = None, window_size: int = 10):
        self.topic_hz = ROSTopicHz(window_size)
        self.topic_delay = ROSTopicDelay(window_size)
        self.topic_bandwidth = ROSTopicBandwidth(window_size)
        T, _, _ = get_topic_class(topic_name)
        assert isinstance(T, type)
        def cb_combined(msg):
            self.topic_hz.callback_hz(msg)
            self.topic_bandwidth.callback(msg)
        sub_any = rospy.Subscriber(topic_name, rospy.AnyMsg, cb_combined)
        rospy.Timer(rospy.Duration(1), self.print_info)
        rospy.Timer(rospy.Duration(0.2), self.reset_by_params)
        self.topic_name = topic_name
        self.rosparam_monitor = RosparamMonitor(monitor_params_prefix)

    def reset_by_params(self, event):
        change_dict = self.rosparam_monitor.detect_change(event)
        if len(change_dict) > 0:
            rospy.logwarn(f"resetting topic monitor due to change in params")
            for param, (old_value, new_value) in change_dict.items():
                rospy.logwarn(f"{param}: {old_value} -> {new_value}")
            self.reset()

    def reset(self):
        self.topic_bandwidth.last_printed_tn = 0
        self.topic_bandwidth.sizes = []
        self.topic_bandwidth.times = []

        self.topic_hz.last_printed_tn = 0
        self.topic_hz.msg_t0 = -1
        self.topic_hz.msg_tn = 0
        self.topic_hz.times = []
        self.topic_hz._last_printed_tn = defaultdict(int)
        self.topic_hz._msg_t0 = defaultdict(lambda: -1)
        self.topic_hz._msg_tn = defaultdict(int)
        self.topic_hz._times = defaultdict(list)

    def print_info(self, event, unit: Literal["B", "KB", "MB"] = "KB"):
        rospy.loginfo(f"hz: {self.get_hz():.2f}, bw: {self.get_bw(unit):.2f} {unit}/s")

    def get_hz(self) -> float:
        ret = self.topic_hz.get_hz()
        if ret is None:
            return -1
        rate, min_delta, max_delta, std_dev, window = ret
        return rate

    def get_bw(self, unit: Literal["B", "KB", "MB"] = "KB") -> float:
        # copied and modified rostopic/ROSTopicBandwidth
        # Copyright (c) 2008, Willow Garage, Inc. All rights reserved.
        if len(self.topic_bandwidth.times) < 2:
            return -1
        with self.topic_bandwidth.lock:
            n = len(self.topic_bandwidth.times)
            tn = time.time()
            t0 = self.topic_bandwidth.times[0]
            
            total = sum(self.topic_bandwidth.sizes)
            bytes_per_s = total / (tn - t0)
            mean = total / n

            max_s = max(self.topic_bandwidth.sizes)
            min_s = min(self.topic_bandwidth.sizes)
        if unit == "B":
            bw, mean, min_s, max_s = [v for v in [bytes_per_s, mean, min_s, max_s]]
        elif unit == "KB":
            bw, mean, min_s, max_s = [v/1000 for v in [bytes_per_s, mean, min_s, max_s]]
        elif unit == "MB":
            bw, mean, min_s, max_s = [v/1000000 for v in [bytes_per_s, mean, min_s, max_s]]
        return bw


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--w", type=int, default=10, help="window size")
    parser.add_argument("--name", type=str, default="/kinect_head/depth_registered/image/compressedDepth", help="topic name")
    parser.add_argument("--params-prefix", type=str, default="/kinect_head/depth_registered/image/compressed", help="prefix of rosparam to monitor")
    args = parser.parse_args()
    rospy.init_node("topic_monitor")
    tm = TopicMonitor(topic_name=args.name, window_size=args.w, monitor_params_prefix=args.params_prefix)
    tm.reset()
    rospy.spin()
