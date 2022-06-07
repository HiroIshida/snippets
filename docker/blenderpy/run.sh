 docker run -it \
        --env="DISPLAY" \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        -v ~/.ssh:/home/h-ishida/.ssh \
        blenderpy
