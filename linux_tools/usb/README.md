## usb関連のdebug
### lsusb
見えているusbデバイスを一覧表示
```
applications@pr1040s:~$ lsusb
Bus 002 Device 001: ID 1d6b:0003 Linux Foundation 3.0 root hub
Bus 001 Device 006: ID 2886:0018 SEEED ReSpeaker 4 Mic Array (UAC1.0)
Bus 001 Device 013: ID 0403:6001 Future Technology Devices International, Ltd FT232 Serial (UART) IC
Bus 001 Device 012: ID 0403:6001 Future Technology Devices International, Ltd FT232 Serial (UART) IC
Bus 001 Device 011: ID 0409:005a NEC Corp. HighSpeed Hub
Bus 001 Device 009: ID 0403:6001 Future Technology Devices International, Ltd FT232 Serial (UART) IC
Bus 001 Device 007: ID 0403:6001 Future Technology Devices International, Ltd FT232 Serial (UART) IC
Bus 001 Device 005: ID 0409:005a NEC Corp. HighSpeed Hub
Bus 001 Device 004: ID 15d1:0000 Hokuyo Data Flex for USB URG-Series USB Driver
Bus 001 Device 003: ID 0a12:0001 Cambridge Silicon Radio, Ltd Bluetooth Dongle (HCI mode)
Bus 001 Device 017: ID 18d1:9302 Google Inc. 
Bus 001 Device 008: ID 15d1:0000 Hokuyo Data Flex for USB URG-Series USB Driver
Bus 001 Device 001: ID 1d6b:0002 Linux Foundation 2.0 root hub
```

### dmesg
カーネルの吐いたメッセージを見る. usb disconnectとかでたら抜けたんだなということがわかる.
```
dmesg
```
試しにマウスを抜き差ししてみるとこんなログが見える
```
[164829.268310] hid-generic 0003:1BCF:08A0.000A: input,hiddev0,hidraw0: USB HID v1.10 Mouse [HID 1bcf:08a0] on usb-0000:0a:00.3-6/input0
[164830.163061] usb 1-1: USB disconnect, device number 6
[164831.332240] usb 1-1: new low-speed USB device number 7 using xhci_hcd
[164831.498242] usb 1-1: New USB device found, idVendor=046d, idProduct=c31c, bcdDevice=64.00
[164831.498249] usb 1-1: New USB device strings: Mfr=1, Product=2, SerialNumber=0
[164831.498252] usb 1-1: Product: USB Keyboard
[164831.498253] usb 1-1: Manufacturer: Logitech
[164831.578595] input: Logitech USB Keyboard as /devices/pci0000:00/0000:00:01.2/0000:01:00.0/0000:02:08.0/0000:0a:00.1/usb1/1-1/1-1:1.0/0003:046D:C31C.000B/input/input35
[164831.636109] hid-generic 0003:046D:C31C.000B: input,hidraw1: USB HID v1.10 Keyboard [Logitech USB Keyboard] on usb-0000:0a:00.1-1/input0
[164831.647162] input: Logitech USB Keyboard Consumer Control as /devices/pci0000:00/0000:00:01.2/0000:01:00.0/0000:02:08.0/0000:0a:00.1/usb1/1-1/1-1:1.1/0003:046D:C31C.000C/input/input36
[164831.704394] input: Logitech USB Keyboard System Control as /devices/pci0000:00/0000:00:01.2/0000:01:00.0/0000:02:08.0/0000:0a:00.1/usb1/1-1/1-1:1.1/0003:046D:C31C.000C/input/input37
[164831.704503] hid-generic 0003:046D:C31C.000C: input,hidraw2: USB HID v1.10 Device [Logitech USB Keyboard] on usb-0000:0a:00.1-1/input1
```

### usb_reset
usb をハードウェア的に抜き差ししなくともreset可能. openni2 にはデフォルトで入っている. 
https://github.com/ros-drivers/openni2_camera/blob/dd6f453ed11e3a68404cfa889530d0d149474645/openni2_camera/src/usb_reset.c#L62
この`usb_script` command は単にusb-deviceのfile名を指定するだけなので, hokuyoでもいける. lsusbでbus名とかcheckする.

```python
import subprocess
import os

p = subprocess.Popen("lsusb", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = p.communicate()
lines = stdout.split("\n")
for line in lines:
    print(line)
print("^^^^^^^^^^^^^^^^^^^^")
# ms_line = [l for l in lines if "ASUS" in l][0]
ms_lines = [l for l in lines if "Hokuyo" in l]
for ms_line in ms_lines:
    print(ms_line)
    tmp = ms_line.split(' ')
    bus_id, device_id = tmp[1], tmp[3]
    print(bus_id)
    full_path = "/dev/bus/usb/" + bus_id + "/" + device_id.rstrip(":")
    print(full_path)
    print('rosrun openni2_camera usb_reset ' + full_path)
    if os.access(full_path, os.W_OK):
        retcode = subprocess.call('rosrun openni2_camera usb_reset ' + full_path, shell=True)
```
