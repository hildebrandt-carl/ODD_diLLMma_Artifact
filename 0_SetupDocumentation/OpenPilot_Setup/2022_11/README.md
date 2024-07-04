# Version xxxx

## Prerequisites

This was tested on a [MSI GE75 Raider 10SE](https://www.msi.com/Laptop/GE75-Raider-10SE/Specification) laptop with a freshly installed version of Ubuntu 20.04 using the default Nvidia Drivers.

The laptop has:
* 10th Generation i7 - 10750H 
* 16GB Ram
* NVIDIA GeForce RTX™ 2060 with 6GB GDDR6

### Basic Programs

I installed basic programs
```bash
sudo apt update -y && sudo apt upgrade -y
sudo apt install git curl wget htop build-essential ffmpeg -y
```

If you haven't already you will also need to initialize git:
```bash
ssh-keygen -t ed25519 -C "<email>"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
```

**Optional Steps**

If you want to enable ssh you can run the following:
```bash
sudo apt install openssh-server
```

If you want to check if your nvidia drivers are installed you can run the following:
```bash
nvidia-smi
```

To install nvidia-drivers your can run
```bash
ubuntu-drivers devices
sudo apt install 
```

## Setting up OpenPilot

Close open pilot and make sure you are using the same commit as I am:
```bash
cd ~
git clone git@github.com:commaai/openpilot.git
cd ~/openpilot 
git checkout a48ec655ac4983145bc93c712ecabac75b886e11
```

Next install it using:
```bash
cd ~/openpilot 
git submodule update --init
tools/ubuntu_setup.sh
```

If you get an error about installing pyenv do the following:
```bash
rm -rf /home/<user>/.pyenv
source ~/.bashrc
tools/ubuntu_setup.sh
```

Then continue the install
```bash
source ~/.bashrc
cd ~/openpilot && poetry shell
USE_WEBCAM=1 scons -j$(nproc) 
```

Test that everything is working by opening the GUI. Note you will need to run through the safety screens. You will also need to check that none of the green keywords in the terminal turn red.
```bash
cd ~/openpilot/tools/sim/
./launch_openpilot.sh
```

## Final Steps Required Before using OpenPilot

I also needed to go into the `model` folder and run
```bash
cd ~/openpilot/selfdrive/modeld/models/
git lfs pull
```

I then installed libzmq from source by first remove libzmq library
```bash
sudo apt purge libzmq* -y
```

Then compile from source:
```bash
cd ~
git clone https://github.com/zeromq/libzmq.git
cd ~/libzmq
sudo apt-get install automake
./autogen.sh
./configure
make
sudo make install
sudo ldconfig
```

## Launching OpenPilot

To run that video through the system you need to use the bridge. Add both the `bridge_video.py` and `run_openpilot.sh` to the folder `tools/sim`. Then install the python packages we will need (make sure you are in their environment using `poetry shell`):

```bash
python3 -m pip install opencv-python
python3 -m pip install h5py
```

<!-- TBD -->

You can test the `bridge_video.py` using:
```bash
# Terminal 1
cd ~/openpilot && poetry shell
cd ~/openpilot/tools/sim/
./launch_openpilot.sh
# Terminal 2
cd ~/openpilot && poetry shell
cd ~/openpilot/tools/sim/
python3 bridge_video.py --filename /Desktop/OpenPilotVideos/a_15.mp4
```

To run multiple videos you can use the following. First make sure there are `mp4` videos in `~/Desktop/Data`. Next run the following:
```bash
# Terminal 1
cd ~/openpilot && poetry shell
cd ~/openpilot/tools/sim/
./run_openpilot.sh
```