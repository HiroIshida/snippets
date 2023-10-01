# windows updateのあとにbootできなくなる問題
https://askubuntu.com/questions/906776/error-failed-to-open-efi-boot-grubx64-efi-dual-booting
live-usbから.efiをコピーする方法はうまくいかなかった. 以下で治った.
```bash
sudo add-apt-repository ppa:yannubuntu/boot-repair && sudo apt update`
sudo apt-get install boot-repair && boot-repair
```
https://askubuntu.com/a/1134107/917108
