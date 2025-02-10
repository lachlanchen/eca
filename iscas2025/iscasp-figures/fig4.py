import pandas as pd
import matplotlib.pyplot as plt

# Prepare the data
data = {
    "Downsample Fraction": [0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1],
    "ECA": [63.99, 77.72, 85.01, 88.82, 91.94, 94.09, 95.14, 96.19, 96.73, 97.26],
    "Logistic Regression": [59.65, 72.89, 80.59, 84.50, 86.62, 88.15, 88.97, 90.23, 91.37, 92.16],
    "LDA": [46.84, 60.88, 63.83, 41.57, 70.20, 80.00, 83.77, 85.99, 86.25, 86.52],
    "QDA": [14.96, 17.07, 19.07, 15.63, 15.75, 15.17, 23.95, 26.19, 18.32, 19.48],
    "SVM": [62.03, 74.33, 81.52, 86.18, 87.99, 89.49, 90.72, 91.66, 92.61, None],
    "Kernel SVM": [45.72, 70.07, 82.80, 88.17, 91.79, 94.14, 95.25, 96.23, 97.30, None]
}

# Create a DataFrame
df_data = pd.DataFrame(data)

# Calculate the downsample rate as 1/fraction
df_data["Downsample Rate"] = 1 / df_data["Downsample Fraction"]

# Plotting
plt.figure(figsize=(8, 6))
for col in df_data.columns[1:-1]:  # Exclude Downsample Fraction and Downsample Rate columns
    linestyle = '--' if col == "ECA" else '-'  # Dashed line for ECA, solid for others
    plt.plot(df_data["Downsample Rate"], df_data[col], label=col, marker='o', linewidth=2.5, linestyle=linestyle)

# Set colors for better distinction
colors = {
    'ECA': 'black',
    'Logistic Regression': 'red',
    'LDA': 'blue',
    'QDA': 'green',
    'SVM': 'purple',
    'Kernel SVM': 'orange'
}
for line in plt.gca().get_lines():
    line.set_color(colors.get(line.get_label(), 'gray'))

# Axis labels and legend
plt.xlabel('Downsample Rate', fontsize=14)
plt.ylabel('Accuracy (%)', fontsize=14)
plt.legend(title='Algorithm', fontsize=10, title_fontsize=11)
plt.xscale('log')  # Set x-axis to log scale

# Customize tick parameters
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Remove grid
plt.grid(False)

plt.tight_layout()

plt.show()

