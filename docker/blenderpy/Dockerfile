FROM kubricdockerhub/blender:latest

# install developper tools:
RUN apt update 
RUN apt install -y build-essential sudo git vim tmux wget

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

RUN echo 'root:root' | chpasswd
RUN echo 'h-ishida:h-ishida' | chpasswd

WORKDIR /home/h-ishida
USER h-ishida

# set up ssh
# https://github.com/IMC3ofC/db2express-c.docker/issues/12
RUN sudo mkdir /var/run/sshd
RUN sudo sed -i 's/\#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
EXPOSE 2233

# setup my own dotfiles (PyYAML is required to intsall)
#RUN sudo apt install python3-pip -y
RUN git clone https://github.com/HiroIshida/dotfiles2.git
RUN pip3 install PyYAML
RUN sudo apt install curl -y
RUN cd dotfiles2 && ./install --light && exec bash
RUN cd dotfiles2/install_scripts && ./install_latest_nodejs.sh 
RUN cd dotfiles2/install_scripts && ./install_vim-plug.sh
