#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt

# Example data
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

# Adjust line thickness based on the y-values
line_widths = [i for i in y]  # Use y-values directly for line thickness

# Plot each segment of the line with different thickness
for i in range(len(x) - 1):
    plt.plot(x[i:i+2], y[i:i+2], color='b', linewidth=line_widths[i])

plt.title("Line Chart with Variable Thickness")
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.show


# In[ ]:




