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
input_range = 5  # Input range of the ADC in volts (e.g., -5V to +5V)
current_source_scale = 10**(-4)  # Current source scaling factor (V/A)
AMPLIFIER_GAIN = 100  # Voltage amplifier gain

# Output directory
output_directory = r"C:\Users\admin\Desktop\Alireza Jozani\Prep3\4'_Vav&Iav"
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
    sampling_rates = [10000, 50000, 200000]  # List of different sampling rates to test
    num_trials = 50  # Number of trials to average
    duration = 0.1  # Duration in seconds (fixed for all sampling rates)

    # Check the ADC gain
    adc_gain = check_adc_gain()
    if adc_gain != 1:
        print("ADC gain is not set to 1. Please configure the ADC gain to 1.")
        return

    resistance_spectra = {}  # Dictionary to store resistance spectra for each sampling rate

    for sampling_rate in sampling_rates:
        num_samples = int(sampling_rate * duration)  # Total number of samples based on fixed duration

        all_adjusted_data_V = []
        all_adjusted_data_I = []

        for i in range(num_trials):
            print(f"Sampling rate {sampling_rate} Hz, Trial {i + 1}/{num_trials}")
            # Acquire data from the ADC
            adc_data = acquire_adc_data(num_samples, sampling_rate, [current_channel, voltage_channel], input_range)
            # Apply the current source scaling factor to convert voltage to current
            current_data = adc_data[0] 
            # Adjust the voltage data for the amplifier gain
            voltage_data = adc_data[1] 
            
            # Remove DC offset from the data and put the data in the list
            current_data -= np.mean(current_data) * current_source_scale
            all_adjusted_data_I.append(current_data) 

            voltage_data -= np.mean(voltage_data) / AMPLIFIER_GAIN
            all_adjusted_data_V.append(voltage_data)

            # Debugging: Print minimum and maximum values
            print(f"Trial {i + 1} Voltage Data Min: {np.min(voltage_data)}, Max: {np.max(voltage_data)}")
            print(f"Trial {i + 1} Current Data Min: {np.min(current_data)}, Max: {np.max(current_data)}")

        # Average the time-domain data
        V_av = np.mean(all_adjusted_data_V, axis=0)
        I_av = np.mean(all_adjusted_data_I, axis=0)
        t_av = np.arange(0, duration, 1/sampling_rate)
        
        # Save the data with the sampling rate as part of the filename
        np.savez_compressed(os.path.join(output_directory, f'data_{sampling_rate}Hz.npz'), V_av=V_av, I_av=I_av, t_av=t_av)
        print(f"Saved V_av and I_av for sampling rate {sampling_rate} Hz")
        
        # Compute noise spectra for V_av and I_av
        freqs_V, power_spectrum_V = compute_noise_spectrum(V_av, sampling_rate)
        freqs_I, power_spectrum_I = compute_noise_spectrum(I_av, sampling_rate)
        
        # Extract positive frequencies starting from the smallest non-zero frequency
        positive_indices_V = freqs_V > 0
        positive_indices_I = freqs_I > 0
        positive_freqs_V = freqs_V[positive_indices_V]
        positive_power_spectrum_V = power_spectrum_V[positive_indices_V]
        positive_freqs_I = freqs_I[positive_indices_I]
        positive_power_spectrum_I = power_spectrum_I[positive_indices_I]
        
        # Calculate the standard deviation (sigma) of the noise
        noise_sigma_current = np.std(I_av)
        noise_sigma_voltage = np.std(V_av)
        
         # Print the results
        print(f"Noise sigma Current (Averaged): {noise_sigma_current:.10f} A")
        print(f"Noise sigma Voltage (Averaged): {noise_sigma_voltage:.10f} V")
        
        
        # Calculate resistance
        resistance = positive_power_spectrum_V / positive_power_spectrum_I
        resistance_spectra[sampling_rate] = (positive_freqs_V, resistance)
        
        # Plot the noise spectrum for V_av and I_av using log-log scale
        plt.figure(figsize=(10, 6))
        plt.loglog(positive_freqs_V, positive_power_spectrum_V, label=f'Voltage Spectrum @ {sampling_rate} Hz, σ={noise_sigma_voltage:.5e} V')
        plt.loglog(positive_freqs_I, positive_power_spectrum_I, label=f'Current Spectrum @ {sampling_rate} Hz, σ={noise_sigma_current:.5e} A')
        plt.title(f'Noise Spectrum of (shorted V_amp&I_scr) V_av and I_av at {sampling_rate} Hz')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_directory, f'noise_spectrum_{sampling_rate}Hz.png'))
        plt.show()

        # Save the noise spectrum data
        np.savez_compressed(os.path.join(output_directory, f'noise_spectrum_{sampling_rate}Hz.npz'), freqs_V=positive_freqs_V, power_spectrum_V=positive_power_spectrum_V, freqs_I=positive_freqs_I, power_spectrum_I=positive_power_spectrum_I)
        print(f"Saved noise spectrum for sampling rate {sampling_rate} Hz")
    
    # Plot resistance spectra
    plt.figure(figsize=(10, 6))
    for sampling_rate, (freqs, resistance) in resistance_spectra.items():
        plt.loglog(freqs, resistance, label=f'Resistance Spectrum @ {sampling_rate} Hz')
    plt.title('Resistance Spectra')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Resistance (Ohms)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'resistance_spectra.png'))
    plt.show()
    print("Saved resistance spectra plot")

if __name__ == "__main__":
    main()
