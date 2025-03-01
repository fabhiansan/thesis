#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import argparse
import os

def load_data(file_path):
    """Load data from the JSON file"""
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return None
    
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not parse {file_path} as JSON.")
        return None

def plot_memory_usage(data, output_file=None):
    """Plot memory usage over time for each GPU"""
    if not data or "data" not in data or not data["data"]:
        print("No data to visualize.")
        return
    
    # Extract timestamps and convert to datetime objects
    timestamps = []
    for entry in data["data"]:
        timestamps.append(datetime.datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S"))
    
    # Get the number of GPUs from the first data point
    num_gpus = len(data["data"][0]["gpus"])
    
    # Create a figure with subplots for each GPU
    fig, axes = plt.subplots(num_gpus, 1, figsize=(12, 4 * num_gpus), sharex=True)
    
    # If there's only one GPU, axes won't be an array
    if num_gpus == 1:
        axes = [axes]
    
    # Plot memory usage for each GPU
    for gpu_idx in range(num_gpus):
        ax = axes[gpu_idx]
        
        # Extract memory usage data for this GPU
        memory_used = []
        memory_total = []
        utilization = []
        
        for entry in data["data"]:
            gpu_data = entry["gpus"][gpu_idx]
            memory_used.append(gpu_data["memory_used_mib"])
            memory_total.append(gpu_data["memory_total_mib"])
            utilization.append(gpu_data["utilization_percentage"])
        
        # Plot memory usage
        ax.plot(timestamps, memory_used, 'b-', label='Memory Used (MiB)')
        ax.plot(timestamps, memory_total, 'r--', label='Total Memory (MiB)')
        
        # Create a second y-axis for utilization
        ax2 = ax.twinx()
        ax2.plot(timestamps, utilization, 'g-.', label='Utilization (%)')
        ax2.set_ylabel('Utilization (%)', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        ax2.set_ylim(0, 105)  # Utilization is a percentage
        
        # Set labels and title
        ax.set_ylabel('Memory (MiB)')
        ax.set_title(f'GPU {gpu_idx} Memory Usage and Utilization')
        
        # Add legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Format the x-axis to show dates nicely
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    
    # Add a title and adjust layout
    fig.suptitle('GPU Memory Usage Over Time', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    # Rotate date labels for better readability
    plt.gcf().autofmt_xdate()
    
    # Save or show the plot
    if output_file:
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize GPU memory usage data')
    parser.add_argument('--input', '-i', default='gpu_memory_usage.json',
                        help='Path to the JSON file containing GPU data (default: gpu_memory_usage.json)')
    parser.add_argument('--output', '-o', help='Path to save the plot (default: show plot)')
    args = parser.parse_args()
    
    data = load_data(args.input)
    if data:
        plot_memory_usage(data, args.output)

if __name__ == "__main__":
    main() 