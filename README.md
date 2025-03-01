# GPU Memory Usage Monitor

This script monitors NVIDIA GPU memory usage every 30 minutes for 3 days and saves the data to a JSON file.

## Requirements

- Python 3.6+
- NVIDIA GPU with drivers installed
- `nvidia-smi` command available in the system

For visualization script:
- matplotlib (`pip install matplotlib`)

## Quick Start

The easiest way to get started is to use the provided shell script:

```bash
chmod +x start_monitoring.sh
./start_monitoring.sh
```

This script will:
1. Check if nvidia-smi and Python are available
2. Install required dependencies
3. Make the Python scripts executable
4. Ask if you want to run the monitoring in the background or foreground

## Manual Usage

### Monitoring Script

1. Make the script executable:
   ```bash
   chmod +x gpu_monitor.py
   ```

2. Run the script:
   ```bash
   ./gpu_monitor.py
   ```

   Or:
   ```bash
   python3 gpu_monitor.py
   ```

3. The script will run for 3 days, checking GPU memory usage every 30 minutes.

4. Data will be saved to `gpu_memory_usage.json` in the same directory.

### Visualization Script

After collecting some data, you can visualize it using the included visualization script:

1. Make the script executable:
   ```bash
   chmod +x visualize_gpu_data.py
   ```

2. Run the script:
   ```bash
   ./visualize_gpu_data.py
   ```

   Or with options:
   ```bash
   python3 visualize_gpu_data.py --input gpu_memory_usage.json --output gpu_plot.png
   ```

## Configuration

You can modify the following variables at the top of the monitoring script:

- `INTERVAL_MINUTES`: Time between checks (default: 30 minutes)
- `DURATION_DAYS`: Total monitoring duration (default: 3 days)
- `OUTPUT_FILE`: Path to the JSON output file (default: "gpu_memory_usage.json")

## Output Format

The script generates a JSON file with the following structure:

```json
{
  "data": [
    {
      "timestamp": "2025-03-01 04:18:24",
      "gpus": [
        {
          "gpu_id": 0,
          "fan_percentage": 30,
          "temperature_c": 21,
          "power_usage_w": 12,
          "power_cap_w": 230,
          "memory_used_mib": 207,
          "memory_total_mib": 24564,
          "utilization_percentage": 0
        },
        {
          "gpu_id": 1,
          "fan_percentage": 30,
          "temperature_c": 20,
          "power_usage_w": 6,
          "power_cap_w": 230,
          "memory_used_mib": 3,
          "memory_total_mib": 24564,
          "utilization_percentage": 0
        }
        // ... more GPUs
      ]
    },
    // ... more data points
  ]
}
```

## Running in Background

To run the script in the background and keep it running even after you log out:

```bash
nohup python3 gpu_monitor.py > gpu_monitor.log 2>&1 &
```

This will start the script in the background and save its output to `gpu_monitor.log`.

## Stopping the Monitoring

If you started the script in the background, you can stop it by finding its process ID and killing it:

```bash
# Find the process ID
ps aux | grep gpu_monitor.py

# Kill the process
kill [PROCESS_ID]
``` 