# Intel Neural Compute Stick 2 Docker Instructions  

This guide describes how to get up and running with the OpenVINO toolkit using docker.

---

## Installing Docker
Use the instructions from Docker's official documentation to install docker on your computer.

Ubuntu:  
https://docs.docker.com/engine/install/ubuntu/

All other OS's:  
https://docs.docker.com/get-docker/

Be sure to run `sudo docker run hello-world` to be sure your docker install was successful.

***Note: You may need to use `sudo docker` to elevate permission on some OS's***

---

## Downloading OpenVINO
The docker images for OpenVINO are located:  
https://hub.docker.com/u/openvino

Use:  
`docker pull openvino/ubuntu20_data_dev:latest`

This command will pull the image from docker hub to your local machine.

---

## Using the container

To run the container the first time run:  
`docker run -it --rm openvino/ubuntu20_data_dev:latest`

This command will enter you into a bash terminal. You should see that your terminal's prompt says `openvino@<hostname>:<path>$`. If you see this you have successfully ran the container and the container has launched an interactive bash terminal.

You will notice that by default the container enters you into `/opt/intel/openvino_<year>.<version>/`. This is where the OpenVINO install is located.

To pass in the device you simply need to pass in the device inside of the `docker run` command:

*For example*:

Use the following command exactly to use OpenVINO with NCS2:  
`docker run -it -u 0 --name myriad --hostname openvinomyriad --device /dev/dri:/dev/dri --device-cgroup-rule='c 189:* rmw' -v /dev/bus/usb:/dev/bus/usb openvino/ubuntu20_data_dev:latest`  

***Note: Using the `--rm` flag when running the container deletes the container after exiting the run.***

Source: https://hub.docker.com/r/openvino/ubuntu20_data_dev

---

## Test the OpenVINO Container
The following link downloads the GoogleNet-v1, optimizes it, and runs it on the Intel NCS2. 

1. `apt update -y && apt upgrade -y`

2. `cd /opt/intel/openvino_2021.*/deployment_tools/open_model_zoo/tools/downloader`

3. `python3 downloader.py --name googlenet-v1 -o ~`

4. `python3 /opt/intel/openvino_2021.*/deployment_tools/model_optimizer/mo.py --input_model ~/public/googlenet-v1/googlenet-v1.caffemodel --data_type FP32 --output_dir ~`

5. `python3 /opt/intel/openvino_2021/deployment_tools/tools/benchmark_tool/benchmark_app.py -m ~/googlenet-v1.xml -d MYRIAD -api async -i /opt/intel/openvino_2021.*/deployment_tools/demo/car.png -b 1`

***NOTE: You will see some long waiting or pinging messages. The model is running on the stick, you may just have to wait.***

Source: https://docs.openvino.ai/latest/openvino_inference_engine_tools_benchmark_tool_README.html

---
## Run and Convert Your Own Model
To see an example of how to implement your own models, convert them for NCS2, and performing inference refer to [lenet.py](docker/lenet.py).  

---

## Runnning in WSL 2
Install usbipd-win on your Windows machine from: https://github.com/dorssel/usbipd-win/releases

In your WSL distro run:  
`sudo apt install linux-tools-5.4.0-77-generic hwdata`  
and  
`sudo update-alternatives --install /usr/local/bin/usbip usbip /usr/lib/linux-tools/5.4.0-77-generic/usbip 20`  

In a Windows Command Prompt run:  
`usbipd wsl list`  
and take note of the Intel NCS2 usb bus id and run:  
`usbipd wsl attach --busid <busid>`

Now you should be able to run `lsusb` in your WSL distro and see the Intel NCS2.

Source: https://devblogs.microsoft.com/commandline/connecting-usb-devices-to-wsl/

---
