# Jetson Nano Setup Guide

This guide will help you set up your Jetson Nano, including downloading the image, burning it to an SD card using Etcher, configuring the system to use extra space for data, and updating and upgrading the system.

## Requirements

- Jetson Nano Developer Kit
- 128GB SD Card (or similar)
- SD Card Reader
- A computer with Etcher installed
- Internet connection

## Step 1: Download the Jetson Nano Image

1. Go to the [NVIDIA Jetson Download Center](https://developer.nvidia.com/embedded/jetson-nano-developer-kit).
2. Download the latest Jetson Nano Developer Kit SD Card Image.
   - Example: `jetson-nano-jp461-sd-card-image.zip`
3. Extract the downloaded `.zip` file to get the `.img` file.
   - Example: `jetson-nano-jp461-sd-card-image.img`

## Step 2: Burn the Image to the SD Card using Etcher

1. Download and install Etcher from [balenaEtcher](https://www.balena.io/etcher/).
2. Insert your SD card into your computer.
3. Open Etcher.
4. Select the Jetson Nano image (`.img` file) you extracted.
5. Select the SD card as the target.
6. Click "Flash!" to start the process.
7. Wait for the process to complete, then safely remove the SD card from your computer.

## Step 3: Configure the System

1. Insert the SD card into the Jetson Nano and power it on.
2. Follow the on-screen instructions to complete the initial setup.
3. When prompted for the System configuration APP partition size, choose the maximum size to utilize all available space for the system.

### Creating an Extra Partition for Data

1. After the initial setup is complete, open a terminal on the Jetson Nano.
2. Identify the remaining unallocated space:
   ```sh
   sudo fdisk -l /dev/mmcblk0
   ```

   - In `fdisk`, follow these steps:
     - Press `n` to create a new partition.
     - Choose `p` for primary partition.
     - Enter a partition number (e.g., `2`).
     - Press `Enter` to accept the default start sector.
     - Press `Enter` again to accept the default end sector (use the remaining space).
     - Press `w` to write the changes and exit `fdisk`.
4. Format the new partition:
   ```sh
   sudo mkfs.ext4 /dev/mmcblk0p2
   ```
5. Create a mount point and mount the new partition:
   ```sh
   sudo mkdir /mnt/newdata
   sudo mount /dev/mmcblk0p2 /mnt/newdata
   ```
6. (Optional) Add the new partition to `/etc/fstab` to mount it automatically at boot:
   ```sh
   echo '/dev/mmcblk0p2 /mnt/newdata ext4 defaults 0 2' | sudo tee -a /etc/fstab
   ```

## Step 4: Update and Upgrade the System

1. Open a terminal on the Jetson Nano.
2. Update the package list:
   ```sh
   sudo apt update
   ```
3. Upgrade all installed packages:
   ```sh
   sudo apt upgrade
   ```

## Step 5: Install Additional Tools and Packages

1. **Install Common Tools and Utilities:**
   ```sh
   sudo apt install build-essential git curl vim
   ```

2. **Install JetPack Components:**
   ```sh
   sudo apt install nvidia-jetpack
   ```

3. **Set Up Swap Space (Optional):**
   - Create a swap file (e.g., 4GB):
     ```sh
     sudo fallocate -l 4G /swapfile
     sudo chmod 600 /swapfile
     sudo mkswap /swapfile
     sudo swapon /swapfile
     ```
   - Add the swap file to `/etc/fstab` to make it permanent:
     ```sh
     echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
     ```

4. **Install Python and Pip:**
   ```sh
   sudo apt install python3 python3-pip
   ```

5. **Set Up a Virtual Environment (Optional):**
   ```sh
   sudo apt install python3-venv
   python3 -m venv myenv
   source myenv/bin/activate
   ```

6. **Install Common Python Packages:**
   ```sh
   pip install numpy pandas matplotlib opencv-python tensorflow keras
   ```

7. **Configure Git (Optional):**
   - Set up Git if you haven't already:
     ```sh
     git config --global user.name "Your Name"
     git config --global user.email "youremail@example.com"
     ```

8. **Enable SSH (Optional):**
   - Enable SSH for remote access:
     ```sh
     sudo systemctl enable ssh
     sudo systemctl start ssh
     ```

