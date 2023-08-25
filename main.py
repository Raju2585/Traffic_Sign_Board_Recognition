import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Data for the bar chart
execution_times = [98,92]  # Execution times for traditional ETL and modern ETL respectively
methods = ['CNN MODEL', 'ALEXNET',]  # Labels for the x-axis
#colors = ['blue', 'green','red']
# Plotting the bar chart
sns.barplot(x=methods, y=execution_times, hue = methods, palette="Set1")

# Adding labels and title
plt.xlabel('Algorithms')
plt.ylabel('Accuracy')
plt.title('Graph Comparison of two models')

plt.yticks(np.arange(0, 110, 10))
plt.xticks()

# Displaying the chart
plt.show()