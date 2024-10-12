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

# Defined the data used
x_data = np.array(range(1, 31))

# Generate x values for plotting the fitted curve
x_fit = np.linspace(1, 30, 30)

fig3, (ax4, ax5) = plt.subplots(1, 2, figsize=(12, 5))

ax4.plot(x_fit, confirmed_normalized, label='Infections', marker='o', linestyle='-')
ax4.set_xlabel('Days since measured n > 4', fontsize=12)
ax4.set_ylabel('n = reported infections/million', fontsize=12)
ax4.set_title('Infection data of first 30 days of contagion since n > 4', fontsize=12, fontweight="bold")
ax4.legend(loc='best')
ax4.set_xticks(range(0, 30, 5))
ax4.set_xticklabels(i for i in range(0, 30, 5))
ax4.set_xlim(left=min(x_fit)-1, right=max(x_fit)+1)
ax4.grid(True)

ax5.plot(x_data, deaths_normalized, label='Deaths', marker='o', linestyle='-', alpha=0.7)
ax5.set_xlabel('Days since measured m > 4', fontsize=12)
ax5.set_ylabel('m = reported deaths/million', fontsize=12)
ax5.set_title('Death data of first 30 days of contagion since m > 4', fontsize=12, fontweight="bold")
ax5.legend(loc='best')
ax5.set_xticks(range(0, 30, 5))
ax5.set_xticklabels(i for i in range(0, 30, 5))
ax5.set_xlim(left=min(x_fit)-1, right=max(x_fit)+1)
ax5.grid(True)

plt.tight_layout()
plt.savefig('covid19_wave_taiwan_raw.png')
plt.show()