import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
import nidaqmx
import time
import pyvisa as visa
import matplotlib.pyplot as plt

# Setup for Noise Generator
rm = visa.ResourceManager()
gen = rm.open_resource('USB0::0x1AB1::0x0642::DG1ZA241701602::INSTR')  # Rigol DG1022 noise generator

def measure_offset():
    """
    Measure the system's offset without the noise signal.
    """
    with nidaqmx.Task() as task:
        task.ai_channels.add_ai_voltage_chan("Dev4/ai4")
        offset_measurements = task.read(number_of_samples_per_channel=300000)
        return np.mean(offset_measurements)

# Measure the system offset with the generator turned off
system_offset = measure_offset()
print('System offset =', system_offset)

# Fixed amplitude and bandwidth settings
amplitude = 3  # Fixed amplitude value
fixed_bandwidth = 200e3  # Fixed bandwidth in Hz
sampling_rates = [50e3, 100e3, 200e3, 300e3, 400e3]  # Example sampling rates



mean_current_values, std_current_values, mean_voltage_values, std_voltage_values = [], [], [], []
skewness_values, kurtosis_values, skewness_values_voltage, kurtosis_values_voltage = [], [], [], []


gen.write('*RST')  # Reset the generator
gen.write('OUTPut1:STATe On')  # Turning on the output of Channel 1



for rate in sampling_rates:
    with nidaqmx.Task() as send_current, nidaqmx.Task() as send_voltage:
        gen.write('FUNCtion NOIS')
        gen.write(f'VOLT {amplitude}')
        gen.write(f'VOLT:OFFS {0}')
        gen.write(f'SOUR1:BWID {fixed_bandwidth}')
        # Setup current and voltage channels
        send_current.ai_channels.add_ai_voltage_chan("Dev4/ai4")
        send_current.timing.cfg_samp_clk_timing(rate)
        send_voltage.ai_channels.add_ai_voltage_chan("Dev4/ai5")
        send_voltage.timing.cfg_samp_clk_timing(rate)

        time.sleep(1)  # Wait for settings to apply

        # Collect data
        current_data = send_current.read(number_of_samples_per_channel=int(rate))
        voltage_data = send_voltage.read(number_of_samples_per_channel=int(rate))

       # Correct for Curent offset
        corrected_current = current_data - abs(system_offset)
        centered_current = corrected_current - np.mean(corrected_current) # Center the data around zero.
#it is to remove any DC offset or systematic bias from the measurement system or the signal source itself.

        
        # Correct for Curent offset
        corrected_voltage = voltage_data - abs(system_offset)
        centered_voltage = corrected_voltage - np.mean(corrected_voltage)
        
        
        # Calculate Statistic for  Current
        current_mean = np.mean(centered_current)
        current_std = np.std(centered_current)
        mean_current_values.append(current_mean)
        std_current_values.append(current_std)
          
        current_skewness = skew(centered_current)
        current_kurtosis = kurtosis(centered_current)
        skewness_values.append(current_skewness)
        kurtosis_values.append(current_kurtosis)
        
        # Calculate Statistic for  voltage
        voltage_mean = np.mean(centered_voltage)
        voltage_std = np.std(centered_voltage)
        mean_voltage_values.append(voltage_mean)
        std_voltage_values.append(voltage_std)
          
        voltage_skewness = skew(centered_voltage)
        voltage_kurtosis = kurtosis(centered_voltage)
        skewness_values_voltage.append(voltage_skewness)
        kurtosis_values_voltage.append(voltage_kurtosis)
        
        time_array = np.arange(len(centered_current)) / rate  # Time in seconds
        time_array1 = np.arange(len(centered_voltage)) / rate  # Time in seconds

        
        fig0 = plt.figure(figsize=(10, 8))
        #plt.rcParams['figure.dpi'] = 300  # or any other value you prefer
        plt.plot(time_array, centered_current, label=f'Sampling rate = {rate / 1e3} kHz', color='black')
        plt.xlabel('Time (s)')
        plt.ylabel('Centered Current (A)')
        plt.title(f'Centered Current vs. Time\nSkewness: {current_skewness:.2f}, Kurtosis: {current_kurtosis:.2f}')
        plt.legend()
        plt.show()
        
        print(f'Sampling Rate: {rate / 1e3} kHz - Mean Current: {current_mean:.5f} A, Std Dev: {current_std:.5f} A, Skewness: {current_skewness:.5f}, Kurtosis: {current_kurtosis:.5f}')
    
        Current_data_path0 = f'C:\\Users\\admin\\Desktop\\Alireza Jozani\\Current_data_{rate}.dat'
        Current_data0 = np.column_stack((time_array, centered_current,))
        np.savetxt(Current_data_path0, Current_data0, delimiter='\t')
#        
        save_path0 = f'C:\\Users\\admin\\Desktop\\Alireza Jozani\\Current_data_{rate}.png' 
        fig0.savefig(save_path0)
        
        # Calculate time array for plotting Voltage data
        fig1 = plt.figure(figsize=(10, 8))
        #plt.rcParams['figure.dpi'] = 300  # or any other value you prefer
        plt.plot(time_array1, centered_voltage, label=f'Sampling rate = {rate / 1e3} kHz', color='black')
        plt.xlabel('Time (s)')
        plt.ylabel('Centered voltage (v)')
        plt.title(f'Centered voltage vs. Time\nSkewness: {current_skewness:.2f}, Kurtosis: {current_kurtosis:.2f}')
        plt.legend()
        plt.show()
        
        voltage_data_path1 = f'C:\\Users\\admin\\Desktop\\Alireza Jozani\\Voltage_data_{rate}.dat'
        Voltage_data1 = np.column_stack((time_array1, centered_voltage,))
        np.savetxt(voltage_data_path1, Voltage_data1, delimiter='\t')
#        
        save_path1 = f'C:\\Users\\admin\\Desktop\\Alireza Jozani\\Voltage_data_{rate}.png' 
        fig1.savefig(save_path1)
    
        # Fourier Transform of Current and plotting
        def dft_current(data):
            N = len(data)
            n = np.arange(N)
            k = n.reshape((N, 1))
            e = np.exp(-2j * np.pi * k * n / N)
            X = np.dot(e, data)
            return X
    
        # Example on a smaller dataset for demonstration
        centered_data_small = centered_current[:1024]  # Taking a smaller subset for manual calculation for the resolution stuff
    
        # Manual DFT computation
        fft_result = dft_current(centered_data_small)
        # Compute the frequencies for the manual FFT result
        fft_frequencies = fft_frequencies = np.fft.fftfreq(len(centered_current), d=1/rate)
        # Compute the magnitude of the manual FFT (since FFT is complex). The magnitude of the
        #Fourier Transform indicates the strength or amplitude of the signal components at each frequency.
        fft_magnitude = np.abs(fft_result)
        
        
    
                # Ensure frequencies are ordered from negative to positive if needed
        # This step is crucial if your frequency array is not already in the correct order
        fft_frequencies_current = np.fft.fftshift(fft_frequencies)
        fft_magnitude_current= np.fft.fftshift(fft_magnitude)
    
        # Now plot the entire spectrum with the adjusted data
        fig2 = plt.figure(figsize=(10, 8))
        plt.plot(fft_frequencies, fft_magnitude, label=f'Sampling Rate = {rate/1e3} kHz')
        plt.title('Magnitude of FFT of Centered Current (Full Spectrum)')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.grid(True)
        plt.legend()
        plt.show()
        
        save_path2 = f'C:\\Users\\admin\\Desktop\\Alireza Jozani\\fft_current_magnitude_{rate}.png' 
        fig2.savefig(save_path2)
    
         # Save FFT data
        fft_data_path2 = f'C:\\Users\\admin\\Desktop\\Alireza Jozani\\fft_current_{rate}.dat'
        fft_data2 = np.column_stack((fft_frequencies_current, fft_magnitude_current))
        np.savetxt(fft_data_path2, fft_data2, delimiter='\t')
        
        
        
         # Fourier Transform of Voltage and plotting
        def dft_voltage(data):
            N = len(data)
            n = np.arange(N)
            k = n.reshape((N, 1))
            e = np.exp(-2j * np.pi * k * n / N)
            X = np.dot(e, data)
            return X
    
        # Example on a smaller dataset for demonstration
        centered_data_small_voltage = centered_voltage[:1024]  # Taking a smaller subset for manual calculation for the resolution stuff
    
        # Manual DFT computation
        fft_result = dft_voltage(centered_data_small_voltage)
        # Compute the frequencies for the manual FFT result
        fft_frequencies = fft_frequencies = np.fft.fftfreq(len(centered_voltage), d=1/rate)
        # Compute the magnitude of the manual FFT (since FFT is complex). The magnitude of the
        #Fourier Transform indicates the strength or amplitude of the signal components at each frequency.
        fft_magnitude = np.abs(fft_result)
        
        
    
        # Ensure frequencies are ordered from negative to positive if needed
        # This step is crucial if your frequency array is not already in the correct order
        fft_frequencies_voltage = np.fft.fftshift(fft_frequencies)
        fft_magnitude_voltage = np.fft.fftshift(fft_magnitude)
    
        # Now plot the entire spectrum with the adjusted data
        fig3= plt.figure(figsize=(10, 8))
        plt.plot(fft_frequencies, fft_magnitude, label=f'Sampling Rate = {rate/1e3} kHz')
        plt.title('Magnitude of FFT of Centered Voltage (Full Spectrum)')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.grid(True)
        plt.legend()
        plt.show()
        
        save_path3 = f'C:\\Users\\admin\\Desktop\\Alireza Jozani\\fft_magnitude_Voltage_{rate}.png' 
        fig3.savefig(save_path3)
    
         # Save FFT data
        fft_data_path_voltage3 = f'C:\\Users\\admin\\Desktop\\Alireza Jozani\\fft_Voltage_{rate}.dat'
        fft_data3 = np.column_stack((fft_frequencies_voltage, fft_magnitude_voltage))
        np.savetxt(fft_data_path_voltage3, fft_data3, delimiter='\t')
#        
        
        
        #Auto Correlation of current Assuming 'current_data' is your signal data for which you want to calculate autocorrelation
        def autocorrelation_current(data1):
            N = len(data1)
            mean_data = np.mean(data1)
            autocorr = np.zeros(N)
            var_data = np.var(data1)  # Calculate the variance of the data for normalization
    
            for delta_t in range(N):
                sum_corr = 0.0
                for t in range(N - delta_t):
                    sum_corr += (data1[t] - mean_data) * (data1[t + delta_t] - mean_data)
                autocorr[delta_t] = sum_corr / N  # Use N for normalization to maintain consistency
    
            # Normalize the autocorrelation by the variance of the data
            autocorr /= var_data
            return autocorr
    
        # Now calculate the normalized manual autocorrelation
        autocorr_result_normalized_current = autocorrelation_current(centered_current)
    
        # Plotting
        fig4 = plt.figure(figsize=(10, 8))
        lags = np.arange(len(autocorr_result_normalized_current))
        plt.plot(lags, autocorr_result_normalized_current, label=f'Sampling Rate = {rate/1e3} kHz', color='teal')
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.title('Normalized Autocorrelation of Centered Current Data')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        save_path4 = f'C:\\Users\\admin\\Desktop\\Alireza Jozani\\AutoCorrCurrent_atRate_{rate}.png'
        fig4.savefig(save_path4)
    
        # # # Save Autocorrelation data
        autocorr_data_path4 = f'C:\\Users\\admin\\Desktop\\Alireza Jozani\\autocorrCurrent_data_{rate}.dat'
        autocorr_data_current4 = np.column_stack((lags, autocorr_result_normalized_current))
        np.savetxt(autocorr_data_path4, autocorr_data_current4, delimiter='\t')

        
          #Auto Correlation of Voltage Assuming 'voltage_data' is your signal data for which you want to calculate autocorrelation
        def autocorrelation_voltage(data):
            N = len(data)
            mean_data = np.mean(data)
            autocorr = np.zeros(N)
            var_data = np.var(data)  # Calculate the variance of the data for normalization
    
            for delta_t in range(N):
                sum_corr = 0.0
                for t in range(N - delta_t):
                    sum_corr += (data[t] - mean_data) * (data[t + delta_t] - mean_data)
                autocorr[delta_t] = sum_corr / N  # Use N for normalization to maintain consistency
    
            # Normalize the autocorrelation by the variance of the data
            autocorr /= var_data
            return autocorr
    
        # Now calculate the normalized manual autocorrelation
        autocorr_result_normalized_voltage = autocorrelation_voltage(centered_voltage)
    
        # Plotting
        fig5 = plt.figure(figsize=(10, 8))
        lags = np.arange(len(autocorr_result_normalized_voltage))
        plt.plot(lags, autocorr_result_normalized_voltage, label=f'Sampling Rate = {rate/1e3} kHz', color='teal')
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.title('Normalized Autocorrelation of Centered voltage Data')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        save_path5 = f'C:\\Users\\admin\\Desktop\\Alireza Jozani\\AutoCorrVoltage_atRate_{rate}.png'
        fig5.savefig(save_path5)
    
        # # # Save Autocorrelation data
        autocorr_data_path5 = f'C:\\Users\\admin\\Desktop\\Alireza Jozani\\autocorrVFoltage_data_{rate}.dat'
        autocorr_data5 = np.column_stack((lags, autocorr_result_normalized_voltage))
        np.savetxt(autocorr_data_path5, autocorr_data5, delimiter='\t')
    
    

# Plotting mean and standard deviation of centered_current
sampling_rates_kHz = [rate / 1e3 for rate in sampling_rates]
fig6 = plt.figure(figsize=(10, 8))
plt.plot(sampling_rates_kHz, mean_current_values, label='Mean Current', marker='o', linestyle='--')
plt.plot(sampling_rates_kHz, std_current_values, label='Standard Deviation of Current', marker='o', linestyle='-')
plt.title('Mean and Standard Deviation of Centered Current vs. Sampling Rate')
plt.xlabel('Sampling Rate (kHz)')
plt.ylabel('Current (A)')
plt.legend()
plt.grid(True)
plt.show()

# Save the figure with two curves inside
save_path6 =f'C:\\Users\\admin\\Desktop\\Alireza Jozani\\mean_and_std_current_vs_sampling_rate.png'
fig6.savefig(save_path6)

data_to_save6 = np.column_stack((sampling_rates_kHz, mean_current_values, std_current_values))
data_file_path6 = 'C:\\Users\\admin\\Desktop\\Alireza Jozani\\mean_and_std_current_vs_sampling_rate.dat'
np.savetxt(data_file_path6, data_to_save6, delimiter='\t', header='Sampling Rate (kHz)\tMean Current (A)\tStandard Deviation of Current (A)', comments='')

# Plotting skewness and kurtosis
sampling_rates_kHz = [rate / 1e3 for rate in sampling_rates]
fig7 = plt.figure(figsize=(10, 8))
plt.plot(sampling_rates_kHz, skewness_values, label='skewnessof Current (third moment)', marker='o', linestyle='--')
plt.plot(sampling_rates_kHz, kurtosis_values, label='kurtosi of current (Fourth Moment)', marker='o', linestyle='-')
plt.title('skewness and kurtosis of Centered Current vs. Sampling Rate')
plt.xlabel('Sampling Rate (kHz)')
plt.ylabel('Current (A)')
plt.legend()
plt.grid(True)
plt.show()

# # Save the figure
save_path7 = f'C:\\Users\\admin\\Desktop\\Alireza Jozani\\skewness_and_kurtosis_current_vs_sampling_rate.png'
fig7.savefig(save_path7)

data_to_save7 = np.column_stack((sampling_rates_kHz, skewness_values, kurtosis_values))
data_file_path7 = 'C:\\Users\\admin\\Desktop\\Alireza Jozani\\skewness_and_kurtosis_current_vs_sampling_rate.dat'
np.savetxt(data_file_path7, data_to_save7, delimiter='\t', header='Sampling Rate (kHz)\tskewness Current (A)\tkurtosis of Current (A)', comments='')


# Plotting mean and standard deviation of centered_voltage
sampling_rates_kHz = [rate / 1e3 for rate in sampling_rates]
fig8 = plt.figure(figsize=(10, 8))
plt.plot(sampling_rates_kHz, mean_voltage_values, label='Mean Voltage', marker='o', linestyle='--')
plt.plot(sampling_rates_kHz, std_voltage_values, label='Standard Deviation of Voltage', marker='o', linestyle='-')
plt.title('Mean and Standard Deviation of Centered voltage vs. Sampling Rate')
plt.xlabel('Sampling Rate (kHz)')
plt.ylabel('Voltage (V)')
plt.legend()
plt.grid(True)
plt.show()

# Save the figure with two curves inside
save_path8 =f'C:\\Users\\admin\\Desktop\\Alireza Jozani\\mean_and_std_voltage_vs_sampling_rate.png'
fig8.savefig(save_path8)

data_to_save8 = np.column_stack((sampling_rates_kHz, mean_voltage_values, std_voltage_values))
data_file_path8 = 'C:\\Users\\admin\\Desktop\\Alireza Jozani\\mean_and_std_voltage_vs_sampling_rate.dat'
np.savetxt(data_file_path8, data_to_save8, delimiter='\t', header='Sampling Rate (kHz)\tMean Voltage (V)\tStandard Deviation of Voltage (V)', comments='')


# Plotting skewness and kurtosis of voltage
sampling_rates_kHz = [rate / 1e3 for rate in sampling_rates]
fig9 = plt.figure(figsize=(10, 8))
plt.plot(sampling_rates_kHz, skewness_values_voltage, label='skewness of voltage (third moment)', marker='o', linestyle='--')
plt.plot(sampling_rates_kHz, kurtosis_values_voltage, label='kurtosis of Voltage (Fourth Moment)', marker='o', linestyle='-')
plt.title('skewness and kurtosis of Centered voltage vs. Sampling Rate')
plt.xlabel('Sampling Rate (kHz)')
plt.ylabel('Voltage (V)')
plt.legend()
plt.grid(True)
plt.show()

# # Save the figure
save_path9 = f'C:\\Users\\admin\\Desktop\\Alireza Jozani\\skewness_and_kurtosis_voltage_vs_sampling_rate.png'
fig9.savefig(save_path9)

data_to_save9 = np.column_stack((sampling_rates_kHz, skewness_values_voltage, kurtosis_values_voltage))
data_file_path9 = 'C:\\Users\\admin\\Desktop\\Alireza Jozani\\skewness_and_kurtosis_voltage_vs_sampling_rate.dat'
np.savetxt(data_file_path9, data_to_save9, delimiter='\t', header='Sampling Rate (kHz)\tskewness Current (A)\tkurtosis of Current (A)', comments='')

        

# Turn off the generator output after the tests
gen.write('OUTPut1:STATe Off')
gen.close()
