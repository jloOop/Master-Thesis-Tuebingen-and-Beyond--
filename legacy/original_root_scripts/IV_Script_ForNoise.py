# -*- coding: utf-8 -*-
"""
Created on Tue May 28 11:01:31 2024

Author: Alireza Jozani
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
import pyvisa as visa
import nidaqmx
import os

# Initialize the signal generator
rm = visa.ResourceManager()
gen = rm.open_resource('USB0::0x1AB1::0x0642::DG1ZA241701602::INSTR')

current_channel = "Dev4/ai7"  # Channel for current measurement
voltage_channel = "Dev4/ai5"  # Channel for voltage measurement

# Signal generator settings
amplitude = 1  # Signal amplitude
input_range = 10  # Input range of the ADC in volts (e.g., -5V to +5V)
current_source_scale = 10**(-4)  # Current source scaling factor (V/A)
AMPLIFIER_GAIN = 10  # Voltage amplifier gain

# Output directory
output_directory = r"C:\Users\admin\Desktop\Alireza Jozani\Prep5\5'_Vav&Iav"
# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Configure the signal generator to produce noise
gen.write('FUNCtion NOIS')
gen.write(f'VOLT {amplitude}')
gen.write(f'VOLT:OFFS 0')

def acquire_adc_data(num_samples, sampling_rate, channels, input_range):
    with nidaqmx.Task() as task:
        # Configure the voltage channels with specified input range
        for channel in channels:
            task.ai_channels.add_ai_voltage_chan(channel, min_val=-input_range, max_val=input_range)
        # Configure the sample clock
        task.timing.cfg_samp_clk_timing(sampling_rate, sample_mode=nidaqmx.constants.AcquisitionType.FINITE, samps_per_chan=num_samples)
        
        # Read the data
        data = task.read(number_of_samples_per_channel=num_samples)
        return np.array(data)

def check_adc_gain():
    adc_gain = 1  # Simulated gain value for demonstration purposes
    return adc_gain    

def compute_noise_spectrum(data, sampling_rate):
    N = len(data)
    fft_result = fft(data)
    power_spectrum = (np.abs(fft_result) ** 2) / (N * sampling_rate)  # Normalize by N and sampling rate
    freqs = fftfreq(N, 1 / sampling_rate)
    return freqs, power_spectrum

def main():
    # Define the number of samples to acquire and the sampling rate
    sampling_rates = [200000]  # List of different sampling rates to test
    num_trials = 50  # Number of trials to average
    duration = 1  # Duration in seconds (fixed for all sampling rates)

    # Check the ADC gain
    adc_gain = check_adc_gain()
    if adc_gain != 1:
        print("ADC gain is not set to 1. Please configure the ADC gain to 1.")
        return

    resistance_spectra_psd = {}  # Dictionary to store resistance spectra for each sampling rate (PSD approach)
    resistance_sigma = {}  # Dictionary to store resistance based on noise sigma

    for sampling_rate in sampling_rates:
        num_samples = int(sampling_rate * duration)  # Total number of samples based on fixed duration

        all_adjusted_data_V = []
        all_adjusted_data_I = []

        for i in range(num_trials):
            print(f"Sampling rate {sampling_rate} Hz, Trial {i + 1}/{num_trials}")
            # Acquire data from the ADC
            adc_data = acquire_adc_data(num_samples, sampling_rate, [current_channel, voltage_channel], input_range)
            # Apply the current source scaling factor to convert voltage to current
            current_data = adc_data[0] * current_source_scale
            # Adjust the voltage data for the amplifier gain
            voltage_data = adc_data[1] / AMPLIFIER_GAIN
            
            # Remove DC offset from the data
            current_data -= np.mean(current_data)
            voltage_data -= np.mean(voltage_data)
            
            # Store the adjusted data
            all_adjusted_data_I.append(current_data)
            all_adjusted_data_V.append(voltage_data)

        # Convert the lists to numpy arrays and save the adjusted data
        all_adjusted_data_V = np.array(all_adjusted_data_V)
        all_adjusted_data_I = np.array(all_adjusted_data_I)

        output_file_path_V = os.path.join(output_directory, f'all_adjusted_data_V_{sampling_rate}Hz_fc=300kHz.txt')
        output_file_path_I = os.path.join(output_directory, f'all_adjusted_data_I_{sampling_rate}Hz_fc=300kHz.txt')
        np.savetxt(output_file_path_V, all_adjusted_data_V, delimiter=',')
        np.savetxt(output_file_path_I, all_adjusted_data_I, delimiter=',')
        print(f"Saved all adjusted voltage and current data for all trials at {sampling_rate} Hz")

        # Compute the noise spectrum for each trial's data and store it
        fft_results_voltage = []
        fft_results_current = []

        for voltage_data, current_data in zip(all_adjusted_data_V, all_adjusted_data_I):
            freqs_voltage, power_spectrum_voltage = compute_noise_spectrum(voltage_data, sampling_rate)
            freqs_current, power_spectrum_current = compute_noise_spectrum(current_data, sampling_rate)
            fft_results_voltage.append(power_spectrum_voltage)
            fft_results_current.append(power_spectrum_current)

        # Convert fft_results to numpy arrays for averaging
        fft_results_voltage = np.array(fft_results_voltage)
        fft_results_current = np.array(fft_results_current)
        avg_fft_current = np.mean(fft_results_current, axis=0)
        avg_fft_voltage = np.mean(fft_results_voltage, axis=0)

        # Extract positive frequencies
        positive_indices = freqs_current > 0
        positive_freqs_current = freqs_current[positive_indices]
        positive_avg_fft_current = avg_fft_current[positive_indices]
        positive_avg_fft_voltage = avg_fft_voltage[positive_indices]
        
        # Calculate the standard deviation (sigma) of the noise from the entire data
        noise_sigma_current = np.std(all_adjusted_data_I)
        noise_sigma_voltage = np.std(all_adjusted_data_V)
        # Print the results
        print(f"Noise sigma Current: {noise_sigma_current:.5f} V")
        print(f"Noise sigma Voltage: {noise_sigma_voltage:.5f} V")
        
        # Calculate resistance using PSD
        resistance_psd = positive_avg_fft_voltage / positive_avg_fft_current
        resistance_spectra_psd[sampling_rate] = (positive_freqs_current, resistance_psd)
        
        # Calculate resistance using noise sigma of noise signals
        resistance_sigma_value = noise_sigma_voltage / noise_sigma_current
        resistance_sigma[sampling_rate] = resistance_sigma_value

        # Plot the averaged FFT power spectrum on a log-log scale
        plt.figure(figsize=(10, 6))
        plt.loglog(positive_freqs_current, positive_avg_fft_current,
                   label=f'Current Spectrum @ {sampling_rate} Hz, σ={noise_sigma_current:.5e} A')
        plt.loglog(positive_freqs_current, positive_avg_fft_voltage,
                   label=f'Voltage Spectrum @ {sampling_rate} Hz, σ={noise_sigma_voltage:.5e} V')
        plt.title(f'Averaged FFT Power Spectrum of a 1K R at {sampling_rate} Hz, Vamp=10, fc=3.5kHz ')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.grid(True, which="both", ls="-")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_directory, f'averaged_fft_power_spectrum_{sampling_rate}H_fc=3.5kHz.png'))
        plt.show()

        # Save the data efficiently using np.savez_compressed
        data_path = os.path.join(output_directory, f'averaged_fft_power_spectrum_data_{sampling_rate}Hz_fc=100kHz.npz')
        np.savez_compressed(data_path, 
                            freqs=positive_freqs_current, 
                            power_spectrum_current=positive_avg_fft_current, 
                            power_spectrum_voltage=positive_avg_fft_voltage, 
                            noise_sigma_current=noise_sigma_current, 
                            noise_sigma_voltage=noise_sigma_voltage)
        print(f"Saved averaged FFT power spectrum data for sampling rate {sampling_rate} Hz")
        
                
    # Plot resistance spectra (PSD approach)
    plt.figure(figsize=(10, 6))
    for sampling_rate, (freqs, resistance) in resistance_spectra_psd.items():
        plt.loglog(freqs, resistance, label=f'Resistance Spectrum (PSD) @ {sampling_rate} Hz')
    plt.title('Resistance Spectra (PSD Approach), Setting Vamp=10_fc=3.5kHz')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Resistance (Ohms)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'resistance_spectra_psd_Setting_Vamp=10_fc=3.5kHz.png'))
    plt.show()
    print("Saved resistance spectra plot (PSD Approach)")

    # Plot resistance calculated using noise sigma
    plt.figure(figsize=(10, 6))
    sampling_rates_list = list(resistance_sigma.keys())
    resistance_sigma_values = list(resistance_sigma.values())
    plt.plot(sampling_rates_list, resistance_sigma_values, marker='o', linestyle='-')
    plt.title('Resistance Calculated Using Noise Sigma, Setting Vamp=10_fc=3.5kHz')
    plt.xlabel('Sampling Rate (Hz)')
    plt.ylabel('Resistance (Ohms)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'resistance_sigma_Setting_Vamp=10_fc=3.5kHz.png'))
    plt.show()
    print("Saved resistance calculated using noise sigma plot")

   

if __name__ == "__main__":
    main()
