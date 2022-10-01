import time
import threading

def func1():
    time.sleep(1)
    print("finish1")

def func2():
    time.sleep(3)
    print("finish2")

if __name__ == '__main__':
    th1 = threading.Thread(target=func1)
    th2 = threading.Thread(target=func2)

    th1.start()
    th2.start()

    thread_list = threading.enumerate()
    thread_list.remove(threading.main_thread())
    for thread in thread_list:
        thread.join()
