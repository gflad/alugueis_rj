# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 22:51:40 2023

@author: gusta
"""

# %% Libs and data
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data = pd.read_csv('web_scraping\\data_to_visualize.csv', encoding='latin-1')

# %% Filtering
# Filtering only for places to rent
data_rent = data[data['predicted_property_type'] == 'Rent']

# Get the top 20 most frequent neighbourhoods
top_20_bairros = data_rent['Bairro'].value_counts().nlargest(20).index

# Filter the data to include only those top 20 'Bairros'
filtered_data = data_rent[data_rent['Bairro'].isin(top_20_bairros)]

# %% 02. Most frequent neighborhoods
# Styling
sns.set_style("whitegrid")

# Plotting the mean rent by neighborhood, bars on the horizontal axis
plt.figure(figsize=(8, 6))

# Now plot the graph
sns.countplot(data=filtered_data, y='Bairro', order=top_20_bairros, color='skyblue')

plt.title("The 20 most frequent neighbourhoods", fontweight="bold")
plt.xlabel("Number of properties for rent")
plt.ylabel(" ")

plt.savefig('web_scraping\\figuras\\02_top20_most_frequent.png', dpi=300, bbox_inches='tight')
plt.show()

# %% 03. Bar plot of median rent

# Calculate mean rent by neighborhood, for the 20 most frequent neighborhoods
median_rent_by_bairro = filtered_data.groupby('Bairro')['aluguel'].median().nlargest(20).reset_index()
mean_rent_by_bairro = filtered_data.groupby('Bairro')['aluguel'].mean().nlargest(20).reset_index()

# Sort the data for better visualization
#mean_rent_by_bairro = mean_rent_by_bairro.sort_values(by='aluguel', ascending=False)

# Styling
sns.set_style("whitegrid")

# Plotting the mean rent by neighborhood, bars on the horizontal axis
plt.figure(figsize=(8, 6))

# Main barplot
sns.barplot(data=median_rent_by_bairro, y='Bairro', x='aluguel', color='skyblue', orient='h')

# Titles and Labels
plt.title("Median Rent by Neighborhood", fontweight="bold")
plt.xlabel("Rent value (in BRL)")
plt.ylabel(" ")

# Limits
plt.xlim([0, 21000])

# Vertical Lines
median_value = filtered_data['aluguel'].median()
mean_value = filtered_data['aluguel'].mean()

# Lighten the colors for vertical lines using RGB tuples
plt.axvline(x=median_value, color=(0.5, 0.5, 1), linestyle='--', linewidth=2)  # Lighter blue
plt.axvline(x=mean_value, color=(1, 0.5, 0.5), linestyle='--', linewidth=2)    # Lighter red


# Annotations
plt.annotate(f'Median: R$ {int(round(median_value))}', xy=(median_value, 19), xytext=(median_value + 300, 19),
             arrowprops=dict(facecolor='blue', arrowstyle='->'), color='blue')
plt.annotate(f'Mean: R$ {int(round(mean_value))}', xy=(mean_value, 18), xytext=(mean_value + 300, 19),
             arrowprops=dict(facecolor='red', arrowstyle='->'), color='red')

# Value labels for bars
for index, value in enumerate(median_rent_by_bairro['aluguel']):
    plt.text(value + 100, index, f'R$ {int(round(value))}', va='center')

plt.savefig('web_scraping\\figuras\\03_median_rent.png', dpi=300, bbox_inches='tight')
plt.show()

# %% 04. Boxplot

# Calculate the median rent for each neighborhood in the filtered data
medians = filtered_data.groupby('Bairro')['aluguel'].median().reset_index()

# Sort the neighborhoods by median rent, from highest to lowest
sorted_bairros = medians.sort_values('aluguel', ascending=False)['Bairro']

# Styling
sns.set_style("whitegrid")

# Outlier properties
flierprops = dict(markerfacecolor='0.99',  # Light grey
                  markersize=3,
                  linestyle='none')

# Create the boxplot
plt.figure(figsize=(8, 6))  # Adjust the figure size

# Use the sorted_bairros for the order parameter
sns.boxplot(data=filtered_data, 
            y='Bairro', 
            x='aluguel', 
            order=sorted_bairros, 
            orient='h', 
            color='skyblue',
            flierprops=flierprops)

# Titles and Labels
plt.title("Distribution of Rent Prices by Neighborhood", fontweight="bold")
plt.xlabel("Rent value (in BRL)")
plt.ylabel(" ")

# Limits
plt.xlim([0, 78000])

# Save the figure
plt.savefig('web_scraping\\figuras\\04_boxplot_rent.png', dpi=300, bbox_inches='tight')

plt.show()


