# Install NVIDIA drivers

## Ubuntu 20.04

* Detect the model of your nvidia graphic card and the recommended driver:
```
$ ubuntu-drivers devices
```
* Install the driver. Should be the recommended one.
```
sudo apt install <driver>
```
* Reboot.

**Note**: to resolve problem with second monitor, install `lightdm`.
