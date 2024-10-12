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
from datetime import datetime
import matplotlib.pyplot as plt

folder_path_all = '/Users/User/Documents/UCLA/Courses/Dynamics and Feedback in Biological and Ecological (279B)/279B HW/279B_HW1/database/data_all'

# Initialize lists to store the data
confirmed_list = []
deaths_list = []
dates_list = []

# Iterate through all csv files in the folder
for file_name in sorted(os.listdir(folder_path_all), key=lambda x: datetime.strptime(x[:-4], "%m-%d-%Y")):
    if file_name.endswith('.csv'):
        # Convert file name to datetime for comparison
        file_date = datetime.strptime(file_name[:-4], "%m-%d-%Y")
        start_date = datetime.strptime("01-22-2020", "%m-%d-%Y")
        
        # Only process files from the desired start date onward
        if file_date >= start_date:
            dates_list.append(file_name[:-4])
            
            # Load the data from the csv file
            file_path = os.path.join(folder_path_all, file_name)
            df = pd.read_csv(file_path)

        # Filter the data for 'Taiwan'
        taiwan_data = df[
            (df['Province/State'].str.contains('Taiwan', na=False) if 'Province/State' in df.columns else False) | 
            (df['Country/Region'].str.contains('Taiwan', na=False) if 'Country/Region' in df.columns else False) | 
            (df['Country_Region'].str.contains('Taiwan', na=False) if 'Country_Region' in df.columns else False)
        ]
        
        # Extract 'Confirmed' and 'Deaths' data
        confirmed = taiwan_data['Confirmed'].values[0] if not taiwan_data.empty else 0
        confirmed = 0 if pd.isna(confirmed) else confirmed
        deaths = taiwan_data['Deaths'].values[0] if not taiwan_data.empty else 0
        deaths = 0 if pd.isna(deaths) else deaths

        # Append the cumulative data to the lists
        confirmed_list.append(confirmed)
        deaths_list.append(deaths)

print('Confirmed List:', confirmed_list, '\n')
print('Deaths List:', deaths_list, '\n')
print('Dated List', dates_list, '\n')

# Create x-axis values as day numbers
days = list(range(1, len(confirmed_list) + 1))
print('days:', days)

# Create two plots in the same figure
fig = plt.figure(figsize=(12, 5))
ax0 = fig.add_subplot(1, 2, 1)
ax1 = fig.add_subplot(1, 2, 2)

# Plot 'Confirmed' and 'Deaths' cases
ax0.plot(days, confirmed_list, label='Infections', linestyle='-', alpha=0.7)
ax1.plot(days, deaths_list, label='Deaths', linestyle='-', alpha=0.7)

# Add labels and title
ax0.set_xlabel('Date', fontsize=12)
ax0.set_ylabel('Number of Infections Reported', fontsize=12)
ax0.set_title('Cumulative Infections', fontsize=16, fontweight="bold")
ax0.legend(loc='best')
ax0.set_xticks([dates_list.index(date) for date in ['01-22-2020', '01-01-2021', '01-01-2022', '01-01-2023'] if date in dates_list])
ax0.set_xticklabels(['Jan 2020', 'Jan 2021', 'Jan 2022', 'Jan 2023'], rotation=45)
ax0.set_xlim(left=min(days)-1, right=max(days))
ax0.grid(True)

ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Number of Deaths Reported', fontsize=12)
ax1.set_title('Cumulative Deaths', fontsize=16, fontweight="bold")
ax1.legend(loc='best')
ax1.set_xticks([dates_list.index(date) for date in ['01-22-2020', '01-01-2021', '01-01-2022', '01-01-2023'] if date in dates_list])
ax1.set_xticklabels(['Jan 2020', 'Jan 2021', 'Jan 2022', 'Jan 2023'], rotation=45)
ax1.set_xlim(left=min(days)-1, right=max(days))
ax1.grid(True)

plt.tight_layout()
plt.savefig('covid19_taiwan.png')
plt.show()