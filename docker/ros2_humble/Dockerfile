FROM ros:humble-ros-core
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    python3 \
    sudo \
    vim-gtk3 \
    git \
    cmake \
    curl \
    wget \
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip install PyYAML
RUN apt-get update && apt install fzf percol -y  # my terminal stuff

# Setup ssh server
RUN apt-get update && apt-get install -y openssh-server net-tools
RUN mkdir /var/run/sshd
RUN sed -i 's/\#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
EXPOSE 22

# User setup
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
