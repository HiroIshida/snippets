FROM osrf/ros:noetic-desktop
RUN apt update 

# install developper tools:
RUN apt install -y build-essential sudo git vim tmux 

# install gui tools: firefox adn xdg-utils for browse files
RUN apt install -y xdg-utils firefox

# networking stuff:
RUN apt install -y openssh-server net-tools

RUN \
  useradd h-ishida && \
  echo "h-ishida ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/h-ishida && \
  chmod 0440 /etc/sudoers.d/h-ishida && \
  mkdir -p /home/h-ishida && \
  chown h-ishida:h-ishida /home/h-ishida && \
  chsh -s /bin/bash h-ishida

# set up ssh
RUN echo 'root:root' | chpasswd
RUN mkdir /var/run/sshd
RUN sed -i 's/\#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
EXPOSE 22

WORKDIR /home/h-ishida
USER h-ishida
CMD /bin/bash
