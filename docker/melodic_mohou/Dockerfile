FROM osrf/ros:melodic-desktop-full
SHELL ["/bin/bash", "-c"]

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
EXPOSE 22

RUN sudo apt install python3-pip -y

# upgrade pip. otherwise source build of opencv-python invoked
RUN pip3 install pip --upgrade

# setup my personal dotfiles (PyYAML is required to intsall)
RUN git clone https://github.com/HiroIshida/dotfiles2.git
RUN pip3 install PyYAML
RUN sudo apt install curl -y
RUN cd dotfiles2 && ./install --light && exec bash

# create catkin_ws
RUN sudo apt update
RUN sudo apt-get install python3-catkin-pkg-modules python3-rospkg-modules python3-venv python3-empy -y
RUN sudo apt-get install ros-melodic-catkin python-wstool -y
RUN sudo apt-get install tree
RUN tree /opt/ros/melodic

# crate workspace
RUN \
    sudo apt-get install python3-catkin-pkg-modules python3-rospkg-modules python3-venv python3-empy -y && \
    sudo apt-get install python-catkin-tools python-wstool ros-melodic-rostest -y
RUN \
    source /opt/ros/melodic/setup.bash && \
    mkdir -p ~/mohou_ws/src && \
    cd ~/mohou_ws/src && \
    git clone https://github.com/HiroIshida/mohou_ros.git && \
    wstool init && \
    wstool merge mohou_ros/rosinstall.melodic && \
    wstool update && \
    rosdep update && \
    rosdep install --from-paths . --ignore-src -y -r && \
    cd ~/mohou_ws && \
    catkin init && \
    catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so && \
    catkin build

RUN \
    source ~/mohou_ws/devel/setup.bash && \
    pip3 install mohou && \
    roscd mohou_ros && sudo pip3 install .

CMD bash
