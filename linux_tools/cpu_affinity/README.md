# isolcpusでOSのprocess割当から特定のcpuを外す.
https://www.ntt-tx.co.jp/column/dpdk_blog/20181127/

Edit `/etc/default/grub` 
```
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash"  # default
GRUB_CMDLINE_LINUX_DEFAULT="isolcpus=1,2,3,5-7"  # changed
```

Then
```
sudo update-grub
sudo reboot
```


