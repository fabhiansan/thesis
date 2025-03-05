#!/usr/bin/env python3
import subprocess
import json
import time
import os
import re
import datetime
from pathlib import Path

# Configuration
INTERVAL_MINUTES = 30
DURATION_DAYS = 3
OUTPUT_FILE = "gpu_memory_usage.json"

# Calculate total iterations
ITERATIONS = (DURATION_DAYS * 24 * 60) // INTERVAL_MINUTES

def run_nvidia_smi():
    """Run nvidia-smi command and return the output"""
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running nvidia-smi: {e}")
        return None
    except FileNotFoundError:
        print("nvidia-smi command not found. Make sure NVIDIA drivers are installed.")
        return None

def parse_memory_usage(output):
    """Parse the memory usage from nvidia-smi output"""
    if not output:
        return None
    
    # Parse the GPU memory usage
    memory_pattern = r"\|\s+(\d+)\s+.*?\s+(\d+)%\s+(\d+)C.*?(\d+)W\s+/\s+(\d+)W\s+\|\s+(\d+)MiB\s+/\s+(\d+)MiB\s+\|\s+(\d+)%"
    matches = re.findall(memory_pattern, output)
    
    gpu_data = []
    for match in matches:
        gpu_id, fan, temp, power_usage, power_cap, mem_used, mem_total, util = match
        gpu_data.append({
            "gpu_id": int(gpu_id),
            "fan_percentage": int(fan),
            "temperature_c": int(temp),
            "power_usage_w": int(power_usage),
            "power_cap_w": int(power_cap),
            "memory_used_mib": int(mem_used),
            "memory_total_mib": int(mem_total),
            "utilization_percentage": int(util)
        })
    
    return gpu_data

def load_existing_data():
    """Load existing data from the JSON file if it exists"""
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Error reading {OUTPUT_FILE}. Creating a new file.")
    return {"data": []}

def save_data(data):
    """Save data to the JSON file"""
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def main():
    print(f"Starting GPU memory monitoring for {DURATION_DAYS} days, checking every {INTERVAL_MINUTES} minutes.")
    print(f"Results will be saved to {OUTPUT_FILE}")
    
    # Load existing data or create new data structure
    data = load_existing_data()
    
    for i in range(ITERATIONS):
        # Get current timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Run nvidia-smi and parse output
        output = run_nvidia_smi()
        gpu_data = parse_memory_usage(output)
        
        if gpu_data:
            # Add new data point
            data_point = {
                "timestamp": timestamp,
                "gpus": gpu_data
            }
            data["data"].append(data_point)
            
            # Save updated data
            save_data(data)
            
            # Print status
            print(f"[{timestamp}] Recorded GPU data (Iteration {i+1}/{ITERATIONS})")
            for gpu in gpu_data:
                print(f"  GPU {gpu['gpu_id']}: {gpu['memory_used_mib']} MiB / {gpu['memory_total_mib']} MiB ({gpu['utilization_percentage']}% util)")
        else:
            print(f"[{timestamp}] Failed to get GPU data (Iteration {i+1}/{ITERATIONS})")
        
        # Wait for next iteration (unless it's the last one)
        if i < ITERATIONS - 1:
            time.sleep(INTERVAL_MINUTES * 60)
    
    print(f"Monitoring completed. Data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main() 