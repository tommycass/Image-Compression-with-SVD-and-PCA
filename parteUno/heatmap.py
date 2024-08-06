import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import matplotlib.pyplot as plt

dir_path = os.path.dirname(os.path.realpath(__file__))
csv_path = os.path.join(dir_path, 'datasets/dataset.csv')
dataset = pd.read_csv(csv_path)
dataset = dataset.drop(columns=['Unnamed: 0'])

# Set the figure size for better visibility
plt.figure(figsize=(15, 10))

# Plotting the heatmap
sns.heatmap(dataset, cmap="viridis", cbar=True)

# Display the heatmap
plt.title('Heatmap del dataset',fontsize=25)
plt.xlabel('Features', fontsize=16.5)
plt.ylabel('Muestras', fontsize=16.5)
plt.xticks([])
plt.yticks([])
plt.show()

