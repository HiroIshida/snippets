## hidデバイスのアクセス権限の設定例.
カーネルが割り当てた, hidraw* という名前のデバイスファイルに対して, 0664 のアクセス権限を設定し, plugdev グループに所属するユーザーがアクセスできるようにする.
```
sudo echo 'KERNEL=="hidraw*", SUBSYSTEM=="hidraw", MODE="0664", GROUP="plugdev"' > /etc/udev/rules.d/99-hidraw-permissions.rules
```
plugdev グループに所属するユーザーを追加する.
```
sudo usermod -aG plugdev $USER
```
すぐさま反映する.
```
newgrp plugdev
```
変更が反映されたか確認する.
```
ls -l /dev/hidraw*
```

## hidデバイスの情報の取得例.
sudo lsusb -vすると, hidデバイスの情報が取得できる. sudoが必要であることに注意.
bInterfaceClassが3のデバイスがhidデバイスである.

```
Bus 001 Device 002: ID 256f:c62e 3Dconnexion SpaceMouse Wireless
Device Descriptor:
  bLength                18
  bDescriptorType         1
  bcdUSB               2.00
  bDeviceClass            0 
  bDeviceSubClass         0 
  bDeviceProtocol         0 
  bMaxPacketSize0        32
  idVendor           0x256f 
  idProduct          0xc62e 
  bcdDevice            4.41
  iManufacturer           1 3Dconnexion
  iProduct                2 SpaceMouse Wireless
  iSerial                 0 
  bNumConfigurations      1
  Configuration Descriptor:
    bLength                 9
    bDescriptorType         2
    wTotalLength       0x0029
    bNumInterfaces          1
    bConfigurationValue     1
    iConfiguration          0 
    bmAttributes         0x80
      (Bus Powered)
    MaxPower              500mA
    Interface Descriptor:
      bLength                 9
      bDescriptorType         4
      bInterfaceNumber        0
      bAlternateSetting       0
      bNumEndpoints           2
      bInterfaceClass         3 Human Interface Device
        ...
