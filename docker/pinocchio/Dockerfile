FROM ubuntu:20.04
RUN apt-get update

RUN echo 'root:root' | chpasswd
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y openssh-server rsync vim-gtk3 python3-pip tmux cmake curl git python3-pip libgl1-mesa-dev sudo
RUN sed -i 's/\#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -i 's/\#PermitEmptyPasswords no/PermitEmptyPasswords yes/' /etc/ssh/sshd_config

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

# copy ssh
COPY --chown=h-ishida .ssh /home/h-ishida/.ssh

# setup dotfiles
RUN git clone git@github.com:HiroIshida/dotfiles2.git
RUN pip install --user PyYAML
RUN cd dotfiles2 && ./install -l
RUN ./dotfiles2/install_scripts/install_vim-plug.sh 
RUN ./dotfiles2/install_scripts/install_latest_nodejs.sh

# setup copilot
RUN git clone https://github.com/github/copilot.vim.git ~/.vim/pack/github/start/copilot.vim
RUN sudo apt install software-properties-common -y
RUN sudo apt update
RUN sudo add-apt-repository ppa:jonathonf/vim -y
RUN sudo apt install vim-gtk3 firefox -y  # firefox for login to github

# install pinocchio
RUN sudo apt install -qqy lsb-release curl
RUN sudo mkdir -p /etc/apt/keyrings
RUN curl http://robotpkg.openrobots.org/packages/debian/robotpkg.asc \
     | sudo tee /etc/apt/keyrings/robotpkg.asc

RUN  echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/robotpkg.asc] http://robotpkg.openrobots.org/packages/debian/pub $(lsb_release -cs) robotpkg" \
     | sudo tee /etc/apt/sources.list.d/robotpkg.list
RUN sudo apt update
RUN sudo apt install -qqy robotpkg-py3*-pinocchio

# setup environment variables (export to bashrc)
RUN echo "export PATH=/opt/openrobots/bin:$PATH" >> ~/.bashrc
RUN echo "export PKG_CONFIG_PATH=/opt/openrobots/lib/pkgconfig:$PKG_CONFIG_PATH" >> ~/.bashrc
RUN echo "export LD_LIBRARY_PATH=/opt/openrobots/lib:$LD_LIBRARY_PATH" >> ~/.bashrc
RUN echo "export PYTHONPATH=/opt/openrobots/lib/python3.10/site-packages:$PYTHONPATH # Adapt your desired python version here" >> ~/.bashrc
RUN echo "export CMAKE_PREFIX_PATH=/opt/openrobots:$CMAKE_PREFIX_PATH" >> ~/.bashrc
