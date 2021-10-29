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

