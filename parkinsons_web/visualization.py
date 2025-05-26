import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("parkinsons/parkinsons.data")

# Drop the 'name' column
data = data.drop(['name'], axis=1)

# Scatter Plot: MDVP:Fo(Hz) vs MDVP:Fhi(Hz)
plt.figure(figsize=(8,6))
sns.scatterplot(x='MDVP:Fo(Hz)', y='MDVP:Fhi(Hz)', hue='status', data=data)
plt.title('Scatter Plot: Average Frequency vs Max Frequency')
plt.xlabel('Average Vocal Frequency (Hz)')
plt.ylabel('Max Vocal Frequency (Hz)')
plt.legend(title='Parkinsons Status')
plt.tight_layout()
plt.savefig("static/scatter_plot.png")
plt.show()
plt.close()  # <- This is important to start a fresh plot

# Box Plot: Distribution of HNR
plt.figure(figsize=(6,4))
sns.boxplot(x='status', y='HNR', data=data)
plt.title('HNR (Harmonic to Noise Ratio) by Status')
plt.xlabel('Status (0 = Healthy, 1 = Parkinson\'s)')
plt.ylabel('HNR')
plt.tight_layout()
plt.savefig("static/box_plot.png")
plt.show()
plt.close()