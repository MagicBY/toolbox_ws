import json
import matplotlib.pyplot as plt

# Load data from the first JSON file
# with open('manipulability_data_Boyu.json', 'r') as f:
with open('manipulability_data_Haviland.json', 'r') as f:
    data_LS_Hqp = json.load(f)

# Load data from the second JSON file
with open('manipulability_data_HMOC.json', 'r') as f:
# with open('manipulability_data_Haviland.json', 'r') as f:    
    data_MMC = json.load(f)

# Extract time and manipulability lists from both datasets
time_list_LS_Hqp = data_LS_Hqp['time_list']
manipulability_list_LS_Hqp = data_LS_Hqp['manipulability_list']

time_list_MMC = data_MMC['time_list']
manipulability_list_MMC = data_MMC['manipulability_list']

# Plotting both datasets on the same plot
plt.figure()
plt.plot(time_list_LS_Hqp, manipulability_list_LS_Hqp, label='HMMC', color='blue')
plt.plot(time_list_MMC, manipulability_list_MMC, label='HMOC', color='red')
plt.xlabel('Time (s)')
plt.ylabel('Manipulability Measure (M)')
plt.title('Manipulability Measure over Time')
plt.legend()
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.show()
