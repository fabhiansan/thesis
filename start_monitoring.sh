#!/bin/bash

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi command not found. Make sure NVIDIA drivers are installed."
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 command not found. Please install Python 3."
    exit 1
fi

# Install dependencies if needed
echo "Installing required Python packages..."
python3 -m pip install -r requirements.txt

# Make scripts executable
chmod +x gpu_monitor.py
chmod +x visualize_gpu_data.py

# Ask if user wants to run in background
read -p "Do you want to run the monitoring script in the background? (y/n): " run_background

if [[ $run_background == "y" || $run_background == "Y" ]]; then
    echo "Starting GPU monitoring in the background..."
    nohup python3 gpu_monitor.py > gpu_monitor.log 2>&1 &
    echo "Monitoring started! Check gpu_monitor.log for output."
    echo "To stop monitoring, find the process ID with: ps aux | grep gpu_monitor.py"
    echo "Then stop it with: kill [PROCESS_ID]"
else
    echo "Starting GPU monitoring in the foreground..."
    python3 gpu_monitor.py
fi 