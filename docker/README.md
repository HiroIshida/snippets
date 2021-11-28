# Use the same IP address while maintaining ssh-ability 
https://stackoverflow.com/questions/62417777/start-docker-container-with-host-network-while-maintaining-the-ability-to-ssh-in
RUN sed -i 's/\(^Port\)/#\1/' /etc/ssh/sshd_config && echo Port 2233 >> /etc/ssh/sshd_config
and 
ssh h-ishida@localhost -p 2233

# How to add  current user to docker group
```
sudo groupadd docker
sudo gpasswd -a $USER docker
sudo systemctl restart docker
exit
```
## autocompletion doesn't work
probably you haven't installed docker-ce-cli because you installed docker via snap. Install docker via apt as the official documentation suggesting.

## container exits immediately after creating docker-compose up
https://github.com/docker/compose/issues/5016
https://qiita.com/rebi/items/580fad9553cd49a07a28
add 
```dockerfile
stdin_open: true
tty: true
```

# If cannot start ssh server
こんなエラーがでる. 
```
h-ishida@4bc3d2c14905:~$ service ssh start 
 * Starting OpenBSD Secure Shell server sshd                                                              Could not load host key: /etc/ssh/ssh_host_rsa_key
Could not load host key: /etc/ssh/ssh_host_ecdsa_key
Could not load host key: /etc/ssh/ssh_host_ed25519_key
```
=> su で行う. 

# 最初にやるかもなこと
https://www.mtioutput.com/entry/2018/09/20/135157
```bash
su
echo "username:password" | chpasswd
```


                                                     
