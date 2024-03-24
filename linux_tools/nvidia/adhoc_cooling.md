## adhoc cooling

```bash
sudo apt install nvidia-settings
vim /etc/X11/xorg.conf
```
and edit
```
Section "Device"
    Identifier     "Device0"
    Driver         "nvidia"
    VendorName     "NVIDIA Corporation"
    Option         "Coolbits" "4"
EndSection
```

then set fan speed to 90% with
```bash
nvidia-settings -a [gpu:0]/GPUFanControlState=1 -a [fan:0]/GPUTargetFanSpeed=90
```

Revert to the default fan control with
```bash
nvidia-settings -a [gpu:0]/GPUFanControlState=0
```
