## Installing OpenPilot

[Install dependencies for Carla for Ubuntu](https://github.com/commaai/openpilot/wiki/CARLA)
```bash
git checkout 
```

Run these commands on a fresh install of Ubuntu 20.04 and it works for me:
Install a few prerequisites:
sudo apt-get update -y
sudo apt-get upgrade -y
sudo apt-get install git -y
sudo apt-get install pipenv -y
Install openpilot/tools/README.md
cd ~
git clone https://github.com/commaai/openpilot.git
cd openpilot 
git submodule update --init
tools/ubuntu_setup.sh
cd ~/openpilot && pipenv shell
scons -u -j$(nproc)
Install Docker
sudo apt update
sudo apt install apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu focal stable"
apt-cache policy docker-ce
sudo apt install docker-ce
sudo systemctl status docker
sudo usermod -aG docker ${USER}
su - ${USER}
groups
Terminal 1:
cd ~/openpilot && pipenv shell
cd ~/openpilot/tools/sim
sudo INSTALL=1 ./start_carla.sh
Terminal 2:
cd ~/openpilot && pipenv shell
cd ~/openpilot/tools/sim
./start_openpilot_docker.sh


[Installing OpenPilot](https://github.com/commaai/openpilot/tree/master/tools)

 
```buildoutcfg
OpenPilot (tag): 6be70a063dfc96b9e9f097f439bdc2e0be54d6d9
openpilot_docker (tag): sha256:35ff99c6004869d97d05a90fa77668f732596d74886f9a5c5fe5003233536b2f
Carla (Version): 0.9.12 (edited)
```

## Controlling OpenPilot within Docker

[commands](https://github.com/commaai/openpilot/blob/master/tools/sim/README.md)

To scroll: press \` and `[`

To exit scroll mode: press `q`

To exit container: `ctrl+d`

To kill process: `ctrl+c`



### OpenPilot Limitations

[Listed here](https://github.com/commaai/openpilot/blob/master/docs/LIMITATIONS.md)

## Troubleshooting

### Stable openpilot version

[Commits to master branch](https://github.com/commaai/openpilot/commits/master?before=f8c81103fc1dc98b0403a89c549947a9777f87ce+140&branch=master)

OpenPilot only tags the latest docker image, but you can still pull from [all untagged docker images](https://github.com/commaai/openpilot/pkgs/container/openpilot-sim/versions?after=100&filters%5Bversion_type%5D=untagged)

### "OpenPilot Unavailable: communication error between processes"
Sometimes this can occur if the OpenPilot docker has been restarted but the Carla docker has not, or vice versa. It is better to restart both docker containers each time you run your code with changes.

If the problem persists, close OpenPilot docker and rebuild. To rebuild, run:

```
cd openpilot/tools
./ubuntu_setup.sh
cd ..
./update_requirements.sh
USE_FRAME_STREAM=1 scons -j$(nproc)
```

### Carla is lagging like crazy
Inside `tools/sim/start_carla.sh` on the last line, add `./CarlaUE4.sh -quality-level=Low` to the end of the docker command. It should look like the following:
`docker run -it --net=host --gpus all carlasim/carla:0.9.7 ./CarlaUE4.sh -quality-level=Low`

### To run low quality
```bash
./start_openpilot_docker.sh --low_quality
```

Troublshooting
Nvidia Graphics Card Driver:
You might run into some issues with your Nvidia driver. To check if this is the issue run the command:
nvidia-smi
If this command fails, you will need to install a new driver.
You can do this though a GUI:
search->Additional Drivers->selected the proprietary, recommended and tested driver
Otherwise you can install a new driver in the terminal using:
ubuntu-drivers devices
Look at the output for the recommended driver and then run

```buildoutcfg
sudo apt install <insert the driver name here>
```

### Can't find your issue anywhere?

Join the [OpenPilot discord](https://discord.comma.ai/).