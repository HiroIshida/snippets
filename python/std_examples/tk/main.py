import time
from display import Display

def run_robot_demo(gui: Display) -> None:
    gui.status("Planning…")
    gui.start_timer()
    time.sleep(2)
    gui.stop_timer("Planned")

    gui.status("Press Proceed to execute", font=("Arial", 120, "bold"))
    gui.enable_proceed()
    gui.wait_proceed()

    gui.status("Executing…")
    time.sleep(3)

    gui.status("Complete!", color="green")
    time.sleep(2)


if __name__ == "__main__":
    with Display() as gui:
        while True:
            gui.wait_start()
            run_robot_demo(gui)
            gui.reset()
