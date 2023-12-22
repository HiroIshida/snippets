import re
import subprocess
from evdev import InputDevice, categorize, ecodes

def find_event_number_for_obinata_key():
    command = "ls -l /dev/input/by-id | grep 'usb-SIGMACHIP_USB_Keyboard-event-kbd'"

    try:
        result = subprocess.check_output(command, shell=True, text=True)
        device_file = result.split()[-1]
    except subprocess.CalledProcessError as e:
        result = "Error occurred: " + str(e)
        device_file = None
    match = re.search(r'event(\d+)', device_file)
    event_number = match.group(1) if match else None
    return event_number


def monitor_keyboard(device_path):
    keyboard = InputDevice(device_path)
    print(f"Monitoring keyboard: {keyboard.name}")
    try:
        for event in keyboard.read_loop():
            if event.type == ecodes.EV_KEY:
                key_event = categorize(event)
                if key_event.keystate == key_event.key_down:
                    print(f"Key pressed: {key_event.keycode}")
    except KeyboardInterrupt:
        print("Stopping keyboard monitoring")


device_path = f"/dev/input/event{find_event_number_for_obinata_key()}"
monitor_keyboard(device_path)
