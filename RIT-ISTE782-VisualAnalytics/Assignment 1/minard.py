import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


"""
Step 1: open datasets
"""
data_troops = pd.read_csv("troops.txt", delim_whitespace=True)
data_temps = pd.read_csv("temps.txt", delim_whitespace=True)
print(data_troops.head(), "\n", data_troops.tail())
print(data_temps.head(), "\n", data_temps.tail())
print(data_troops.describe(), "\n", data_temps.describe())
print(data_troops.shape, "\n", data_temps.shape)

"""
Step 2: draw the first graph (troops)
"""
fig = plt.figure(figsize=(10, 6))
gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
ax_1 = fig.add_subplot(gs[0])
ax_2 = fig.add_subplot(gs[1])

advance = data_troops["direction"].values == "A"
plot_data = data_troops[advance]
plot_data = [plot_data[:16], plot_data[16:22], plot_data[22:]]
ax_1.set_ylim(53, 57)
ax_1.set_xlim(22.5, 39)
for i in range(3):
    line_widths = plot_data[i]["survivors"].values / 10000
    for j in range(len(plot_data[i]) - 1):
        ax_1.plot(plot_data[i]["long"].values[j:j + 2], plot_data[i]["lat"].values[j:j + 2], color="red",
                     linewidth=line_widths[j])

# plt.figure(figsize=(10, 5))
# plt.ylim(53, 57)
# plt.xlim(22.5, 39)
# for i in range(3):
#     line_widths = plot_data[i]["survivors"].values / 10000
#     for j in range(len(plot_data[i]) - 1):
#         plt.plot(plot_data[i]["long"].values[j:j + 2], plot_data[i]["lat"].values[j:j + 2], color="red",
#                  linewidth=line_widths[j])
retreat = data_troops["direction"].values == "R"
plot_data = data_troops[retreat]
plot_data = [plot_data[:19], plot_data[19:23], plot_data[23:]]
for i in range(3):
    line_widths = plot_data[i]["survivors"].values / 10000
    for j in range(len(plot_data[i]) - 1):
        ax_1.plot(plot_data[i]["long"].values[j:j + 2], plot_data[i]["lat"].values[j:j + 2], color="black",
                 linewidth=line_widths[j])
data_cities = pd.read_csv("cities.txt", sep="\s+")
for i in range(len(data_cities)):
    ax_1.text(data_cities["long"].values[i], data_cities["lat"].values[i], data_cities["city"].values[i], fontsize=8,
             color="grey", ha="right")

"""
Step 3: draw the second graph (temperature)
"""
plot_data = data_temps.sort_values("long")
ax_2.set_ylim(-32, 2)
ax_2.plot(plot_data["long"], plot_data["temp"], marker="o", color="blue")
for i in range(len(plot_data)):
    ax_2.text(plot_data["long"].values[i], plot_data["temp"].values[i], f'{plot_data["month"].values[i]}{plot_data["day"].values[i]}', fontsize=9,
             color="grey", ha='right')
plt.savefig("minard.png")
plt.show()

"""
Step 4: Add labels to both graphs
"""
"""
Step 5: Combine the graphs in a FacetGrid
"""
