import myext
import time
import signal

def handler(signum, frame):
    raise TimeoutError("Timeout")

def my_callback():
    pass

signal.signal(signal.SIGALRM, handler)
signal.alarm(5)
try:
    myext.start_infinite_loop(my_callback)
except TimeoutError:
    print("Timeout")
