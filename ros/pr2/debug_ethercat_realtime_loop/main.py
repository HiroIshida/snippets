from pathlib import Path
import argparse
import time
import matplotlib.pyplot as plt
import psutil
import rospy
from diagnostic_msgs.msg import DiagnosticArray
import json
from threading import Lock


class Monitor:

    def __init__(self):
        topic = "/diagnostics"
        sub = rospy.Subscriber(topic, DiagnosticArray, self.callback)
        n_cpu = psutil.cpu_count(logical=False)
        self.json_dict = {"n_cpu": n_cpu}
        self.lock = Lock()
        self.ts = time.time()

    def push(self, key, value):
        if key not in self.json_dict:
            self.json_dict[key] = []
        self.json_dict[key].append(float(value))
        rospy.loginfo(f"{key}: {value}")

    def callback(self, msg: DiagnosticArray):
        with self.lock:
            name = "Realtime Control Loop"
            for status in msg.status:
                if status.name == name:
                    for value in status.values:
                        if value.key == "Avg EtherCAT roundtrip (us)":
                            self.push("current_time", time.time() - self.ts)
                            self.push("roundtrip_time", value.value)
                            cpu_percentages = psutil.cpu_percent(interval=1, percpu=True)
                            for i, cpu_percentage in enumerate(cpu_percentages):
                                self.push(f"cpu{i}", cpu_percentage)

    def save(self, file_name: str):
        with self.lock:
            with open(file_name, "w") as f:
                json.dump(self.json_dict, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Plot the data")
    args = parser.parse_args()

    file_name = "data.json"
    if args.plot:
        with open(file_name, "r") as f:
            data = json.load(f)
        fig, ax = plt.subplots()
        print(data["roundtrip_time"])
        ax.plot(data["current_time"], data["roundtrip_time"], label="Roundtrip Time", marker="o", linestyle="-")
        for i in range(data["n_cpu"]):
            ax.plot(data["current_time"], data[f"cpu{i}"], label=f"CPU {i}")
        ax.legend()
        plt.show()
    else:
        if Path(file_name).exists():
            input(f"{file_name} already exists. To overwrite, press [Enter] => ")
        rospy.init_node("listener")
        monitor = Monitor()
        rospy.spin()
        monitor.save(file_name)
        rospy.loginfo(f"Saved data to {file_name}")
