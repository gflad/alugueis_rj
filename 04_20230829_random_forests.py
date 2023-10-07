# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 14:47:33 2023

@author: gusta
"""

# Libraries and data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
# from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

data_rf = pd.read_csv(
    'web_scraping\\data_mun_rf.csv', encoding='latin-1')

# %% Splitting into train/test and doing a first run

# For simplicity, removing variables that wont be used (several missing values)
data_rf = data_rf[['aluguel', 'Condomínio', 'IPTU', 'tamanho_total', 'lat',
                   'long', 'CEP', 'Bairro', 'new_property_type']]

# Let's train a Random Forest model using only the labeled observations about the type of the property
# (if it is for sale or for rent), and then use the trained model to predict the labels of the 'Uncertain' observations

# First step: filter the data
labeled_data = data_rf[data_rf['new_property_type'] != 'Uncertain']
unlabeled_data = data_rf[data_rf['new_property_type'] == 'Uncertain']

# Train/Test split
features_response = ['aluguel', 'Condomínio', 'IPTU', 'tamanho_total', 'lat', 'long']
X = labeled_data[features_response]
y = labeled_data['new_property_type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

# Training the model with arbitrary values for n_estimators and max_depth
rf_model = RandomForestClassifier(
    n_estimators=20,
    max_depth=4,
    random_state=24)


# %% Cross Validating

# Create a parameter grid to search the numbers of trees and maximum depth
rf_params = {'n_estimators': list(range(10, 110, 10)),
             'max_depth': [3, 6, 9, 12]}


# Grid search cross-validation object using the parameter grid that was just created
cv_rf = GridSearchCV(rf_model, param_grid=rf_params, scoring='roc_auc',
                     n_jobs=None, refit=True, cv=4, verbose=1,
                     pre_dispatch=None, error_score=np.nan, return_train_score=True)

# Fit the cross-validation object
cv_rf.fit(X_train, y_train)

# Putting the cross-validation results into dataframe
cv_rf_results_df = pd.DataFrame(cv_rf.cv_results_)
cv_rf_results_df

# ideal set of hyperparameters to use
cv_rf.best_params_

# additionally, create a DataFrame of the feature names and importance, and then show it sorted by importance
feat_imp_df = pd.DataFrame({
    'Feature name': features_response,
    'Importance': cv_rf.best_estimator_.feature_importances_
})

feat_imp_df.sort_values('Importance', ascending=False)

# %% Second run
# Running the model again for the ideal set of hyperparameters {'max_depth': 12, 'n_estimators': 30}
# 3. Train the Model
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(
    n_estimators=30,  # numero de arvores
    max_depth=12,  # tamanho maximo de cada arvore
    random_state=24)

rf_model.fit(X_train, y_train)


# Classifying the 'Uncertain' observations
X_unlabeled = unlabeled_data[features_response]
unlabeled_data = unlabeled_data.copy()
unlabeled_data['predicted_property_type'] = rf_model.predict(X_unlabeled)

# Joining the unlabeled with the labeled
labeled_data['predicted_property_type'] = labeled_data['new_property_type']

# Row binding labeled and unlabeled
data = pd.concat([labeled_data, unlabeled_data])

# Dropping old classification
data = data.drop(columns=['new_property_type'])

# Exporting
data.to_csv('web_scraping\\data_to_visualize.csv', index=False, encoding='latin-1')

# %% Quick data visualization with the final dataset

# Filtering only for places to rent
data_rent = data[data['predicted_property_type'] == 'Rent']

# Get the top 20 most frequent neighbourhoods
top_20_bairros = data_rent['Bairro'].value_counts().nlargest(20).index

# Filter the data to include only those top 20 'Bairros'
filtered_data = data_rent[data_rent['Bairro'].isin(top_20_bairros)]

# Now plot the graph
sns.countplot(data=filtered_data, x='Bairro', order=top_20_bairros, color='lightblue')
plt.title("Distribution of properties for rent by top 20 neighbourhoods")
plt.ylabel("Number of properties announced for rent")
plt.xticks(rotation=45)
plt.show()

# %% Calculating the median rent by neighborhood, and then plotting

# Calculate mean rent by neighborhood, for the 20 most frequent neighborhoods
median_rent_by_bairro = filtered_data.groupby('Bairro')['aluguel'].median().nlargest(20).reset_index()
mean_rent_by_bairro = filtered_data.groupby('Bairro')['aluguel'].mean().nlargest(20).reset_index()

# Sort the data for better visualization
#mean_rent_by_bairro = mean_rent_by_bairro.sort_values(by='aluguel', ascending=False)

# Plotting the mean rent by neighborhood, bars on the horizontal axis
plt.figure(figsize=(12, 10))
sns.barplot(data=median_rent_by_bairro, y='Bairro', x='aluguel', color='grey', orient='h')
plt.title("Median Rent by Neighborhood")
plt.xlabel("Rent value (in BRL)")
plt.ylabel("Neighborhood")
plt.xlim([0, 21000])

# Add a red, dashed vertical line with the median
median_value = filtered_data['aluguel'].median()
plt.axvline(x=median_value, color='red', linestyle='--')

# Blue dashed line with the mean
mean_value = filtered_data['aluguel'].mean()
plt.axvline(x=mean_value, color='blue', linestyle='--')

# Add text labels next to each bar
for index, value in enumerate(median_rent_by_bairro['aluguel']):
    plt.text(value, index, f'R$ {int(round(value))}')

# Add text near the vertical line
plt.text(median_value + 300, 19, f'Median: R$ {int(round(median_value))}', color='red')
plt.text(mean_value + 300, 19, f'Mean: R$ {int(round(mean_value))}', color='blue')


plt.show()


#%%
# Import necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Styling
sns.set_style("whitegrid")

# Plotting the mean rent by neighborhood, bars on the horizontal axis
plt.figure(figsize=(8, 6))

# Main barplot
sns.barplot(data=median_rent_by_bairro, y='Bairro', x='aluguel', color='skyblue', orient='h')

# Titles and Labels
plt.title("Median Rent by Neighborhood")
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

plt.savefig('web_scraping\\figuras\\median_rent.png', dpi=300, bbox_inches='tight')
plt.show()
