## usb関連のdebug

### usb memory が認識されなくなった (filesystemが壊れている.)
lsblkでusbがblock deviceとして認識されているか確認. 抜き差しすると以下が変化した.
```
...
sdc           8:32   1  58.2G  0 disk 
└─sdc1        8:33   1  58.2G  0 part 
```

```
sudo fsck dev/sdc
```
をして, 古いpartision sdc1を削除して, 新しいpartisionを作成する. 基本的にはdefaultでenterを押していけばよい (e.g. primary). そしてフォーマットする.
```
sudo mkfs.vfat /dev/sdc1
```

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

### USB の packet 監視
1番bus(1u)のusb packetを監視する.
```
sudo cat /sys/kernel/debug/usb/usbmon/1u
```
こんな感じでBulk Out/In のpacketがタイムフレームと転送方式とともに表示される (CAN通信のケース)
USB通信は1.interrupt, 2.bulk, 3.control, 4.isochronous の4つのtransfer typeがある. Bi, BoのBはbulkを表す.
S, Cはそれぞれsubmit/completeを表す.
```
ffff9ff50ab86480 988832077 S Bo:1:013:1 -115 22 = 74303032 38374646 46374646 30303030 30303746 460d
ffff9ff50ab86480 988832186 C Bo:1:013:1 0 22 >
ffff9ff50ab86480 988832206 S Bo:1:013:1 -115 22 = 74303033 38374646 46374646 30303030 30303746 460d
ffff9ff50ab86480 988832241 C Bo:1:013:1 0 22 >
ffff9ff50ab86480 988832250 S Bo:1:013:1 -115 22 = 74303034 38374646 46374646 30303030 30303746 460d
ffff9ff50ab86480 988832304 C Bo:1:013:1 0 22 >
ffff9ff50ab86240 988833216 C Bi:1:013:1 0 22 = 74303132 38313238 30303437 46463746 46314131 410d
ffff9ff50ab86240 988833218 S Bi:1:013:1 -115 128 <
ffff9ff50ab87680 988833269 C Bi:1:013:1 0 22 = 74303133 38313337 46333137 46453746 45314231 390d
ffff9ff50ab87680 988833271 S Bi:1:013:1 -115 128 <
ffff9ff50ab86900 988833374 C Bi:1:013:1 0 22 = 74303134 38313438 31394137 46463746 46314131 410d
ffff9ff50ab86900 988833376 S Bi:1:013:1 -115 128 <
ffff9ff50ab87a40 988833499 C Bi:1:013:1 0 22 = 74303135 38313538 31334237 46463746 45314131 410d
```

ちなみにマウス通信の場合のpacketは次のようで, interruptモードのinのみであることがわかる.
```
h-ishida@umejuice:~$ sudo cat /sys/kernel/debug/usb/usbmon/1u
[sudo] password for h-ishida: 
ffff9ff4f5453e00 2412708810 C Ii:1:009:1 0:2 6 = 01000050 ff00
ffff9ff4f5453e00 2412708847 S Ii:1:009:1 -115:2 6 <
ffff9ff4f5453e00 2412710842 C Ii:1:009:1 0:2 6 = 010000a0 ff00
ffff9ff4f5453e00 2412710858 S Ii:1:009:1 -115:2 6 <
ffff9ff4f5453e00 2412718801 C Ii:1:009:1 0:2 6 = 01000050 ff00
ffff9ff4f5453e00 2412718817 S Ii:1:009:1 -115:2 6 <
ffff9ff4f5453e00 2412726790 C Ii:1:009:1 0:2 6 = 01000040 ff00
```
