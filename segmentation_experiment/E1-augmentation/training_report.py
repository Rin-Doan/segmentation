"""
Training Report Generation Module
Handles generation of training efficiency and convergence reports
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
import torch


def get_emissions_attr(emissions_obj, attr_name, default=None):
    """Safely get attribute from EmissionsData object"""
    if emissions_obj is None:
        return default
    try:
        return getattr(emissions_obj, attr_name, default)
    except (AttributeError, TypeError):
        return default


def process_emissions_data(emissions_tracker):
    """
    Stop energy tracking and extract emissions data
    
    Args:
        emissions_tracker: CodeCarbon EmissionsTracker object
        
    Returns:
        emissions_data: EmissionsData object or None
    """
    emissions_data = None
    if emissions_tracker is not None:
        try:
            emissions_tracker.stop()
            emissions_data = emissions_tracker.final_emissions_data
            if emissions_data:
                energy = get_emissions_attr(emissions_data, 'energy_consumed_kWh', 0)
                co2 = get_emissions_attr(emissions_data, 'emissions', 0)
                country = get_emissions_attr(emissions_data, 'country_name', 'Unknown')
                
                print(f"\n🌍 ENERGY/CARBON FOOTPRINT:")
                print(f"   Energy consumed: {energy:.4f} kWh")
                print(f"   CO2 equivalent: {co2:.4f} kg CO2eq")
                print(f"   Location: {country}")
        except Exception as e:
            print(f"⚠️  Warning: Failed to get emissions data: {e}")
            emissions_data = None
    
    return emissions_data


def save_training_efficiency_report(
    total_training_time,
    avg_epoch_time,
    avg_ram,
    avg_gpu,
    emissions_data,
    csv_file='training_efficiency.csv'
):
    """
    Save training efficiency report to CSV
    
    Args:
        total_training_time: Total training time in seconds
        avg_epoch_time: Average epoch time in seconds
        avg_ram: Average RAM usage in GB
        avg_gpu: Average GPU memory usage in GB
        emissions_data: EmissionsData object or None
        csv_file: Output CSV file path
    """
    training_efficiency_data = {
        'Total Training Time (hours)': [total_training_time / 3600],
        'Total Training Time (second)': [total_training_time],
        'Average Epoch Time(s)': [avg_epoch_time],
        'Average RAM (GB)': [avg_ram],
        'Average GPU Memory (GB)': [avg_gpu],
        'Energy Consumed (kWh)': [get_emissions_attr(emissions_data, 'energy_consumed_kWh')],
        'CO2 Emissions (kg)': [get_emissions_attr(emissions_data, 'emissions')],
    }
    
    if os.path.exists(csv_file):
        df_existing = pd.read_csv(csv_file)
        df_new = pd.DataFrame(training_efficiency_data)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(csv_file, index=False)
        print(f"✓ Training efficiency report appended to '{csv_file}' (Total runs: {len(df_combined)})")
    else:
        df = pd.DataFrame(training_efficiency_data)
        df.to_csv(csv_file, index=False)
        print(f"✓ Training efficiency report saved to '{csv_file}' (New file created)")


def save_convergence_report(
    convergence_epoch,
    epoch_times,
    memory_usage_history,
    total_training_time,
    emissions_data,
    csv_file='convergence.csv'
):
    """
    Save convergence report to CSV (only if convergence occurred)
    
    Args:
        convergence_epoch: Epoch number when convergence was detected (None if not converged)
        epoch_times: List of epoch times in seconds
        memory_usage_history: List of memory usage dictionaries per epoch
        total_training_time: Total training time in seconds
        emissions_data: EmissionsData object or None
        csv_file: Output CSV file path
    """
    if not convergence_epoch:
        print("⚠️  No convergence detected - skipping convergence.csv")
        return
    
    convergence_time_total = sum(epoch_times[:convergence_epoch])
    convergence_avg_epoch_time = np.mean(epoch_times[:convergence_epoch])
    
    # Calculate average RAM/GPU up to convergence point
    convergence_memory_history = memory_usage_history[:convergence_epoch]
    convergence_avg_ram = np.mean([m['ram_gb'] for m in convergence_memory_history])
    convergence_avg_gpu = np.mean([m['gpu_allocated_gb'] for m in convergence_memory_history]) if torch.cuda.is_available() else 0
    
    # Estimate energy/carbon up to convergence (proportional to time)
    convergence_energy = None
    convergence_co2 = None
    if emissions_data:
        total_energy = get_emissions_attr(emissions_data, 'energy_consumed_kWh', 0)
        total_co2 = get_emissions_attr(emissions_data, 'emissions', 0)
        if total_energy and total_training_time > 0:
            # Proportional energy consumption up to convergence
            convergence_energy = total_energy * (convergence_time_total / total_training_time)
            convergence_co2 = total_co2 * (convergence_time_total / total_training_time) if total_co2 else None
    
    convergence_data = {
        'Convergence Epoch': [convergence_epoch],
        'Total Training Time (hours)': [convergence_time_total / 3600],
        'Total Training Time (second)': [convergence_time_total],
        'Average Epoch Time(s)': [convergence_avg_epoch_time],
        'Average RAM (GB)': [convergence_avg_ram],
        'Average GPU Memory (GB)': [convergence_avg_gpu],
        'Energy Consumed (kWh)': [convergence_energy],
        'CO2 Emissions (kg)': [convergence_co2],
    }
    
    if os.path.exists(csv_file):
        df_existing = pd.read_csv(csv_file)
        df_new = pd.DataFrame(convergence_data)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(csv_file, index=False)
        print(f"✓ Convergence report appended to '{csv_file}' (Total runs: {len(df_combined)})")
    else:
        df = pd.DataFrame(convergence_data)
        df.to_csv(csv_file, index=False)
        print(f"✓ Convergence report saved to '{csv_file}' (New file created)")


def generate_training_reports(
    epoch_times,
    memory_usage_history,
    convergence_epoch,
    emissions_tracker,
    csv_dir='training_reports'
):
    """
    Generate all training reports (efficiency and convergence)
    
    Args:
        epoch_times: List of epoch times in seconds
        memory_usage_history: List of memory usage dictionaries per epoch
        convergence_epoch: Epoch number when convergence was detected (None if not converged)
        emissions_tracker: CodeCarbon EmissionsTracker object
        csv_dir: Directory to save CSV files 
    """
    # Create directory if it doesn't exist
    os.makedirs(csv_dir, exist_ok=True)
    
    # Process emissions data
    emissions_data = process_emissions_data(emissions_tracker)
    
    # Calculate training metrics
    total_training_time = sum(epoch_times)
    avg_epoch_time = np.mean(epoch_times)
    avg_ram = np.mean([m['ram_gb'] for m in memory_usage_history])
    avg_gpu = np.mean([m['gpu_allocated_gb'] for m in memory_usage_history]) if torch.cuda.is_available() else 0
    
    # Save training efficiency report
    training_efficiency_file = os.path.join(csv_dir, 'training_efficiency.csv')
    save_training_efficiency_report(
        total_training_time,
        avg_epoch_time,
        avg_ram,
        avg_gpu,
        emissions_data,
        training_efficiency_file
    )
    
    # Save convergence report
    convergence_file = os.path.join(csv_dir, 'convergence.csv')
    save_convergence_report(
        convergence_epoch,
        epoch_times,
        memory_usage_history,
        total_training_time,
        emissions_data,
        convergence_file
    )
    
    return emissions_data

