# Complete Guide: Using the Neural Compute Stick 2
## 1) Setting up virtual environment
In order to get the Neural Compute Stick 2 working, it is strongly recommended that you use a fresh install of Ubuntu on a virtual environment. (I couldn't get Movidius working on my main machine even with Ubuntu). 

 1. Download VirtualBox on your host machine
`sudo apt-get install virtualbox`
 2. Download Ubuntu disk image found [here](https://ubuntu.com/download/desktop) (I'm using Ubuntu 20.04.3 LTS) 
 3. Install and set up Ubuntu on a VirtualBox machine (you can follow this [guide](https://brb.nci.nih.gov/seqtools/installUbuntu.html) for help)
 4. Shutdown the virtual machine
 5. Install the VirtualBox expansion pack for your version of VirtualBox so we can use USB 3.0 drivers (you can follow this [guide](https://www.nakivo.com/blog/how-to-install-virtualbox-extension-pack/#:~:text=Open%20Launchpad,%20run%20VirtualBox,%20then,VirtualBox%20site%20%28Oracle_VM_VirtualBox_Extension_Pack-6.0.))

Now your virtual machine should be set up with USB 3.0 capabilities. But in order for the virtual machine to "see" the Neural Compute Stick 2, we must configure the VirtualMachine.

 1. In VirtualBox select the virtual machine you have created and click "settings"
 2. Click on "USB"
 3. Click "create USB filter", edit the filter and enter the following
 `Vendor ID: 03e7`
 `Product ID: 2485`
 4.  Click "create USB filter again", edit the filter and enter the following
 `Vendor ID: 03e7`
 `Product ID: f63b`
 5. Set the drivers to USB 3.0
 6. Start your virtual machine, and type `lsusb`in the terminal
 7. Make sure you see `ID 03e7:2485 Intel Movidius MyriadX`in the output

## 2) Setting up OpenVino on the virtual machine

Follow the getting started guide on Intels website ([here](https://www.intel.com/content/www/us/en/developer/articles/guide/get-started-with-neural-compute-stick.html))

If you have an error with PyYAML, enter the following
`sudo -H pip3 install --ignore-installed PyYAML`

At this point, all of the demos in
`~/intel/openvino_2021/deployment_tools/demo`
Should work and run on `MYRIAD`

## 3) Running custom models on the Neural Compute Stick 2
Follow this aptly named guide: [The battle to run my custom network on a Movidius / Myriad Compute Stick](https://medium.com/analytics-vidhya/the-battle-to-run-my-custom-network-on-a-movidius-myriad-compute-stick-c7c01fb64126)
Or use the code provided in this repo

For keras2onnx we must use Tensorflow 2.2.0
`pip3 install tensorflow==2.2.0`
