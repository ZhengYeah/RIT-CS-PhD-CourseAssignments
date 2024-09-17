#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt

# Example data
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

# Create the plot
plt.plot(x, y, marker='o', color='b')

# Add labels for each point
for i in range(len(x)):
    plt.text(x[i], y[i], f'({x[i]}, {y[i]})', fontsize=9, ha='right')

plt.title("Line Chart with Labels for Each Value")
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.show()


# In[ ]:




