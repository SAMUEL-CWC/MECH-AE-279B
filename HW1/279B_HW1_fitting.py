#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module Name: 279B_HW1.py
Description: This script is used for UCLA MECH&AE 279B HW1.
Author: Samuel Chien
Date Created: 2024-10-05
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from scipy.optimize import curve_fit

folder_path_2020 = '/Users/User/Documents/UCLA/Courses/Dynamics and Feedback in Biological and Ecological (279B)/279B HW/279B_HW1/database/data_2020'
folder_path_2021 = '/Users/User/Documents/UCLA/Courses/Dynamics and Feedback in Biological and Ecological (279B)/279B HW/279B_HW1/database/data_2021'

# Initialize lists to store the data
confirmed_wave_list =[]
confirmed_a_list = []
deaths_wave_list = []
recovered_wave_list = []
active_wave_list = []

# Iterate through csv files from 0318 to 0416 in 2020 (n>4)
for file_name in sorted(os.listdir(folder_path_2020)):
    if file_name.endswith('.csv') and '03-20-2020.csv' <= file_name <= '04-18-2020.csv':
        # Load the data from the csv file
        file_path_wave = os.path.join(folder_path_2020, file_name)
        df = pd.read_csv(file_path_wave)

        # Filter the data for 'Taiwan'
        taiwan_data_wave = df[
            (df['Province/State'].str.contains('Taiwan', na=False) if 'Province/State' in df.columns else False) | 
            (df['Country/Region'].str.contains('Taiwan', na=False) if 'Country/Region' in df.columns else False) | 
            (df['Country_Region'].str.contains('Taiwan', na=False) if 'Country_Region' in df.columns else False)
        ]
        
        # Extract 'Confirmed' and 'Deaths' data
        confirmed_wave = taiwan_data_wave['Confirmed'].values[0] if not taiwan_data_wave.empty else 0
        confirmed_wave= 0 if pd.isna(confirmed_wave) else confirmed_wave
        # deaths_wave = taiwan_data_wave['Deaths'].values[0] if not taiwan_data_wave.empty else 0
        # deaths_wave = 0 if pd.isna(deaths_wave) else deaths_wave
        recovered_wave = taiwan_data_wave['Recovered'].values[0] if not taiwan_data_wave.empty else 0
        recovered_wave = 0 if pd.isna(recovered_wave) else recovered_wave
        active_wave = confirmed_wave - recovered_wave
        active_wave = 0 if active_wave < 0 else active_wave

        # Append the cumulative data to the lists
        confirmed_wave_list.append(confirmed_wave)
        # deaths_wave_list.append(deaths_wave)
        recovered_wave_list.append(recovered_wave)
        active_wave_list.append(active_wave)

for file_name in sorted(os.listdir(folder_path_2021)):
    if file_name.endswith('.csv') and '05-29-2021.csv' <= file_name <= '06-27-2021.csv':
        # Load the data from the csv file
        file_path_wave = os.path.join(folder_path_2021, file_name)
        df = pd.read_csv(file_path_wave)

        # Filter the data for 'Taiwan'
        taiwan_data_wave = df[
            (df['Province/State'].str.contains('Taiwan', na=False) if 'Province/State' in df.columns else False) | 
            (df['Country/Region'].str.contains('Taiwan', na=False) if 'Country/Region' in df.columns else False) | 
            (df['Country_Region'].str.contains('Taiwan', na=False) if 'Country_Region' in df.columns else False)
        ]
        
        # Extract 'Deaths' data
        deaths_wave = taiwan_data_wave['Deaths'].values[0] if not taiwan_data_wave.empty else 0
        deaths_wave = 0 if pd.isna(deaths_wave) else deaths_wave

        # Append the cumulative deaths data to the list
        deaths_wave_list.append(deaths_wave)

print('Confirmed List:', confirmed_wave_list , '\n')
print('Deaths List:', deaths_wave_list, '\n')
print('Recovered List:', recovered_wave_list, '\n')
print('Active List:', active_wave_list, '\n')

# Create a new figure with 'Confirmed', 'Deaths', and 'Active' data (from 10th to 39th element)
fig2 = plt.figure(figsize=(12, 5))
ax2 = fig2.add_subplot(1, 2, 1)
ax3 = fig2.add_subplot(1, 2, 2)

# Normalized the data to #cases/million
days_wave = list(range(1, len(confirmed_wave_list)))
confirmed_normalized = np.array([x / 23.56 for x in confirmed_wave_list])
deaths_normalized = np.array([x / 23.56 for x in deaths_wave_list])
recovered_normalized = np.array([x / 23.56 for x in recovered_wave_list])
active_normalized = np.array([x / 23.56 for x in active_wave_list])

print('Confirmed Wave:', confirmed_normalized, '\n')
print('Deaths Wave:', deaths_normalized, '\n')
print('Recoverd Wave', recovered_normalized, '\n')
print('Active Wave:', active_normalized, '\n')

# Define the exponential model
def exponential_model(x, a, b):
    return a * np.exp(b * x)

# Define the Gompertz model
def gompertz_model(x, a, b, c):
    return a * np.exp(b * (1-np.exp(-c * x)))

# Define the Logistic model
def logistic_model(x, a, b, c):
    return a * (b / (b + (a - b) * np.exp(-c * x)))

# Defined the data used
x_data = np.array(range(1, 31))

# Fit the exponential model to the data
confirmed_params_exp, _ = curve_fit(exponential_model, x_data, confirmed_normalized, p0=[confirmed_normalized[0], 0.1])
deaths_params_exp, _ = curve_fit(exponential_model, x_data, deaths_normalized, p0=[deaths_normalized[0], 0.1])
active_params_exp, _ = curve_fit(exponential_model, x_data, active_normalized, p0=[active_normalized[0], 0.1])

# Fit the Gompertz model to the data
confirmed_params_g, _ = curve_fit(gompertz_model, x_data, confirmed_normalized, p0=[confirmed_normalized.max(), -1, 0.1])
deaths_params_g, _ = curve_fit(gompertz_model, x_data, deaths_normalized, p0=[deaths_normalized.max(), -1, 0.1])
active_params_g, _ = curve_fit(gompertz_model, x_data, active_normalized, p0=[active_normalized.max(), -1, 0.1])

# Fit the Logistic model to the data
confirmed_params_log, _ = curve_fit(logistic_model, x_data, confirmed_normalized, p0=[confirmed_normalized.max(), confirmed_normalized.min(), 0.1])
deaths_params_log, _ = curve_fit(logistic_model, x_data, deaths_normalized, p0=[deaths_normalized.max(), deaths_normalized.min(), 0.1])
active_params_log, _ = curve_fit(logistic_model, x_data, active_normalized, p0=[active_normalized.max(), active_normalized.min(), 0.1])

# Print the parameters for each dataset
print('Confirmed_Parameters_Exp:', confirmed_params_exp, '\n')
print('Deaths Parameters_Exp:', deaths_params_exp, '\n')
print('Active Parameters_Exp:', active_params_exp, '\n')

print('Confirmed_Parameters_G:', confirmed_params_g, '\n')
print('Deaths_Parameters_G:', deaths_params_g, '\n')
print('Active_Parameters_G:', active_params_g, '\n')

print('Confirmed_Parameters_Log:', confirmed_params_log, '\n')
print('Deaths_Parameters_Log:', deaths_params_log, '\n')
print('Active_Parameters_Log:', active_params_log, '\n')

# Generate x values for plotting the fitted curve
x_fit = np.linspace(1, 30, 30)

# Plot the confirmed data and the fitted model
ax2.plot(x_data, confirmed_normalized, label='Infections', marker='o', linestyle='-', alpha=0.7, color='gray')
ax2.plot(x_fit, exponential_model(x_fit, *confirmed_params_exp), label='Exponential Fit', color='red')
ax2.plot(x_fit, gompertz_model(x_fit, *confirmed_params_g), label='Gompertz Fit', color='blue')
ax2.plot(x_fit, logistic_model(x_fit, *confirmed_params_log), label='Logistic Fit', color='green')
ax2.set_xlabel('Days since measured n > 4', fontsize=12)
ax2.set_ylabel('n = reported infections/million', fontsize=12)
ax2.set_title('Infection data of first 30 days of contagion since n > 4', fontsize=12, fontweight="bold")
ax2.legend(loc='best')
ax2.set_xticks(range(0, 30, 5))
ax2.set_xticklabels(i for i in range(0, 30, 5))
ax2.set_xlim(left=min(x_fit)-1, right=max(x_fit)+0.5)
ax2.grid(True)

# Plot the deaths data and the fitted model
ax3.plot(x_data, deaths_normalized, label='Deaths', marker='o', linestyle='-', alpha=0.7, color='gray')
ax3.plot(x_fit, exponential_model(x_fit, *deaths_params_exp), label='Exponential Fit', color='red')
ax3.plot(x_fit, gompertz_model(x_fit, *deaths_params_g), label='Gompertz Fit', color='blue')
ax3.plot(x_fit, logistic_model(x_fit, *deaths_params_log), label='Logistic Fit', color='green')
ax3.set_xlabel('Days since measured m > 4', fontsize=12)
ax3.set_ylabel('m = reported cumulative deaths/million', fontsize=12)
ax3.set_title('Death data of first 30 days of contagion since m > 4', fontsize=12, fontweight="bold")
ax3.legend(loc='best')
ax3.set_xticks(range(0, 30, 5))
ax3.set_xticklabels(i for i in range(0, 30, 5))
ax3.set_xlim(left=min(x_fit)-1, right=max(x_fit)+0.5)
ax3.grid(True)

plt.tight_layout()
plt.savefig('covid19_wave_taiwan.png')
plt.show()

# Figure 3 - Error plots
fig4, (ax6, ax7) = plt.subplots(1, 2, figsize=(12, 5))
# Calculate and plot the error for confirmed data
confirmed_error_exp = np.divide((confirmed_normalized - exponential_model(x_data, *confirmed_params_exp)), confirmed_normalized) * 100
confirmed_error_g = np.divide((confirmed_normalized - gompertz_model(x_data, *confirmed_params_g)), confirmed_normalized) * 100
confirmed_error_log = np.divide((confirmed_normalized - logistic_model(x_data, *confirmed_params_log)),  confirmed_normalized) * 100
ax6.plot(x_data, confirmed_error_exp, label='Exponential Fitting Error', marker='o', linestyle='-', color='red')
ax6.plot(x_data, confirmed_error_g, label='Gompertz Fitting Error', marker='o', linestyle='-', color='blue')
ax6.plot(x_data, confirmed_error_log, label='Logistic Fitting Error', marker='o', linestyle='-', color='green')
ax6.axhline(y=11, ls="--", color="k")
ax6.axhline(y=-11, ls="--", color="k")
ax6.set_xlabel('Days since measured n > 4', fontsize=12)
ax6.set_ylabel('Error (%)', fontsize=12)
ax6.set_title('Fitting error of each model for infections', fontsize=14, fontweight="bold")
ax6.set_ylim(-15, 15)
ax6.legend(loc='best')

# Calculate and plot the error for deaths data
deaths_error_exp = np.divide((deaths_normalized - exponential_model(x_data, *deaths_params_exp)), deaths_normalized) * 100
deaths_error_g = np.divide((deaths_normalized - gompertz_model(x_data, *deaths_params_g)), deaths_normalized) * 100
deaths_error_log = np.divide((deaths_normalized - logistic_model(x_data, *deaths_params_log)), deaths_normalized) * 100
ax7.plot(x_data, deaths_error_exp, label='Exponential Fitting Error', marker='o', linestyle='-', color='red')
ax7.plot(x_data, deaths_error_g, label='Gompertz Fitting Error', marker='o', linestyle='-', color='blue')
ax7.plot(x_data, deaths_error_log, label='Logistic Fitting Error', marker='o', linestyle='-', color='green')
ax7.axhline(y=-20, ls="--", color="k")
ax7.axhline(y=20, ls="--", color="k")
ax7.set_xlabel('Days since measured m > 4', fontsize=12)
ax7.set_ylabel('Error (%)', fontsize=12)
ax7.set_title('Fitting error of each model for deaths', fontsize=14, fontweight="bold")
ax7.set_ylim(-25, 25)
ax7.legend(loc='best')

print('confirmed_error_exp', confirmed_error_exp, '\n')
print('confirmed_error_g', confirmed_error_g, '\n')
print('confirmed_error_log', confirmed_error_log, '\n')

print('deaths_error_exp', deaths_error_exp, '\n')
print('deaths_error_g', deaths_error_g, '\n')
print('deaths_error_log', deaths_error_log, '\n')

plt.tight_layout()
plt.savefig('fitting_error.png')
plt.show()

# Function to calculate RMSE
def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

confirmed_rmse_g = calculate_rmse(confirmed_normalized, gompertz_model(x_data, *confirmed_params_g))
confirmed_rmse_log = calculate_rmse(confirmed_normalized, logistic_model(x_data, *confirmed_params_log))

deaths_rmse_g = calculate_rmse(deaths_normalized, gompertz_model(x_data, *deaths_params_g))
deaths_rmse_log = calculate_rmse(deaths_normalized, logistic_model(x_data, *deaths_params_log))

print('Confirmed RMSE Gompertz', confirmed_rmse_g, '\n')
print('Confirmed RMSE Logistic', confirmed_rmse_log, '\n')
print('Deaths RMSE Gompertz', deaths_rmse_g, '\n')
print('Deaths RMSE Logistic', deaths_rmse_log, '\n')

# Plot the recoverd data, active data and the fitted model
fig5, (ax8, ax9, ax10) = plt.subplots(1, 3, figsize=(15, 5))

ax8.plot(x_data, recovered_normalized, label='Recovered', marker='o', linestyle='-', alpha=0.7, color='gray')
ax8.set_xlabel('Days since measured p > 4', fontsize=10)
ax8.set_ylabel('reported recoveries/million', fontsize=10)
ax8.set_title('Recoverd data of first 30 days of contagion since p > 4', fontsize=10, fontweight="bold")
ax8.legend(loc='best')
ax8.set_xticks(range(0, 30, 5))
ax8.set_xticklabels(i for i in range(0, 30, 5))
ax8.set_xlim(left=min(x_data)-1, right=max(x_data)+0.5)
ax8.grid(True)

ax9.plot(x_data, active_normalized, label='Active', marker='o', linestyle='-', alpha=0.7, color='gray')
ax9.plot(x_fit, exponential_model(x_fit, *active_params_exp), label='Exponential Fit', color='red')
ax9.plot(x_fit, gompertz_model(x_fit, *active_params_g), label='Gompertz Fit', color='blue')
ax9.plot(x_fit, logistic_model(x_fit, *active_params_log), label='Logistic Fit', color='green')
ax9.set_xlabel('Days since measured p > 4', fontsize=10)
ax9.set_ylabel('p = reported active infections/million', fontsize=10)
ax9.set_title('Active infections of first 30 days of contagion since p > 4', fontsize=10, fontweight="bold")
ax9.legend(loc='best')
ax9.set_xticks(range(0, 30, 5))
ax9.set_xticklabels(i for i in range(0, 30, 5))
ax9.set_xlim(left=min(x_data)-1, right=max(x_data)+0.5)
ax9.grid(True)

active_error_exp = np.divide((active_normalized - exponential_model(x_data, *active_params_exp)), active_normalized) * 100
active_error_g = np.divide((active_normalized - gompertz_model(x_data, *active_params_g)), active_normalized) * 100
active_error_log = np.divide((active_normalized - logistic_model(x_data, *active_params_log)), active_normalized) * 100
ax10.plot(x_data, active_error_exp, label='Exponential Fitting Error', marker='o', linestyle='-', color='red')
ax10.plot(x_data, active_error_g, label='Gompertz Fitting Error', marker='o', linestyle='-', color='blue')
ax10.plot(x_data, active_error_log, label='Logistic Fitting Error', marker='o', linestyle='-', color='green')
ax10.set_xlabel('Days since measured p > 4', fontsize=10)
ax10.set_ylabel('Error (%)', fontsize=10)
ax10.set_title('Fitting error of each model for active infections', fontsize=10, fontweight="bold")
ax10.legend(loc='best')
ax10.set_xticks(range(0, 30, 5))
ax10.set_xticklabels(i for i in range(0, 30, 5))
ax10.set_xlim(left=min(x_data)-1, right=max(x_data)+0.5)
ax10.grid(True)

plt.tight_layout()
plt.savefig('active.png')
plt.show()

print('active_error_exp', active_error_exp, '\n')
print('active_error_g', active_error_g, '\n')
print('active_error_log', active_error_log, '\n')