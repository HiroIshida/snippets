import os
import datetime 
import re
import subprocess
from pydub import AudioSegment
import simpleaudio as sa
from evdev import InputDevice, categorize, ecodes
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


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


def remove_chars_loop(input_string, chars_to_remove):
    result = ""
    for char in input_string:
        if char not in chars_to_remove:
            result += char
    return result


def decode(input_string):
    input_string = remove_chars_loop(input_string, ["a", "b", "c"])
    if len(input_string) != 32:
        raise ValueError("Input string should be exactly 32 characters long")
    tmp = [input_string[i : i + 4] for i in range(0, len(input_string), 4)]
    return "".join([tmp[0], tmp[2], tmp[4], tmp[6]])


def send_email(subject, message, to_email):
    password = decode(os.environ.get("EMAIL_PASS"))
    my_address = "spitfire.docomo@gmail.com"
    to_address = to_email  # recipient address

    msg = MIMEMultipart()
    msg["From"] = my_address
    msg["To"] = to_address
    msg["Subject"] = subject

    msg.attach(MIMEText(message, "plain"))
    server = smtplib.SMTP("smtp.gmail.com: 587")
    server.starttls()
    server.login(my_address, password)
    server.sendmail(my_address, to_address, msg.as_string())
    server.quit()


device_path = f"/dev/input/event{find_event_number_for_obinata_key()}"
keyboard = InputDevice(device_path)
print(f"Monitoring keyboard: {keyboard.name}")
try:
    for event in keyboard.read_loop():
        if event.type == ecodes.EV_KEY:
            key_event = categorize(event)
            if key_event.keystate == key_event.key_down:
                current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                message = f"{current_date}: key pressed"

                audio = AudioSegment.from_file("./kawaii.mp3", format="mp3")
                raw_audio_data = audio.raw_data
                sample_rate = audio.frame_rate
                play_obj = sa.play_buffer(raw_audio_data, num_channels=audio.channels, bytes_per_sample=audio.sample_width, sample_rate=sample_rate)
                play_obj.wait_done()
                dong_addr = os.environ.get("DONG_EMAIL_ADDR")
                send_email("dong xu so cute!!", message, dong_addr)


except KeyboardInterrupt:
    print("Stopping keyboard monitoring")
