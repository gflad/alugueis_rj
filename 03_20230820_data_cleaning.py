# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 23:22:37 2023

@author: gusta
"""

%reset - f  # noqa

# %% bibliotecas e dados

from sklearn.ensemble import RandomForestClassifier  # noqa 
import seaborn as sns  # noqa 
import matplotlib.pyplot as plt  # noqa 
import pandas as pd
import numpy as np
import geobr
import shapely
import descartes
pd.set_option('display.float_format', '{:.2f}'.format)


data = pd.read_csv(
    'web_scraping\\olx_20230820.csv', encoding='latin-1')

# %% visao geral dos dados e estrategia
data.info()
data.head()

# 'Acomoda' e 'Características' so tem nulos, podemos remover:
data = data.drop(['Acomoda', 'Características'], axis=1)

# duplicatas
data.duplicated().sum()

# removendo duplicatas mas mantendo uma
data = data.drop_duplicates(keep='first')


# %% Transformando variaveis em numericas
# %%% Aluguel: removendo pontos usados
data['aluguel'] = data['aluguel'].replace(r'\.', '', regex=True).astype(float)

# %%% 'Condomínio', 'IPTU', 'Aluguel + Condomínio': removendo R$, pontos e transformando em numerico
for col in ['Condomínio', 'IPTU', 'Aluguel + Condomínio']:
    data[col] = data[col].str.replace("R$ ", "", regex=False)
    data[col] = data[col].replace(r'\.', '', regex=True).astype(float)

# %%% 'Tamanho', 'Área útil', 'Área construída': removendo m² e transformando em numerico
for col in ['Tamanho', 'Área útil', 'Área construída']:
    data.loc[pd.notna(data[col]), col] = data.loc[pd.notna(
        data[col]), col].str.slice(stop=-2)
    data[col] = data[col].astype(float)

# tamanho_total
data['tamanho_total'] = data[['Tamanho', 'Área útil', 'Área construída']].sum(axis=1, min_count=1)

# Removing the columns
data = data.drop(['Tamanho', 'Área útil', 'Área construída'], axis=1)

# %%% Banheiros e Vagas na garagem
data['Banheiros'].value_counts()
data['Vagas na garagem'].value_counts()

for col in ['Banheiros', 'Vagas na garagem']:
    data[col] = data[col].str.slice(stop=1).astype(float)

del col

# %% Classificando aluguel ou venda

# Passo 1: identificar duplicatas
sort_columns = ['CEP', 'Município', 'Bairro', 'Condomínio', 'IPTU', 'tamanho_total']
data = data.sort_values(by=sort_columns)
data['is_duplicate'] = data.duplicated(subset=sort_columns, keep=False)

# Passo 2: contar duplicatas
data['is_duplicate'].sum()

# conta dentro de cada grupo
data[data['is_duplicate']].groupby(sort_columns).size().reset_index(name='counts').head()

# quantas observacoes se repetem N vezes? (maioria so repete 2x)
data['repeat_count'] = data.groupby(sort_columns)['aluguel'].transform('count')

# verificando
data['repeat_count'].value_counts()

# remove observacoes que repetem mais de 3x
data = data[data['repeat_count'] <= 2]

# para as observacoes que repetem 2x comparar o valor do aluguel
# coluna indica quantas vezes o aluguel de uma e maior que da outra


# Filter observations that repeat exactly twice
# data_twice = data[data['repeat_count'] == 2].sort_values(by=sort_columns)

# Initialize new columns
data['aluguel_ratio'] = np.nan
data['new_property_type'] = ''

# Identify duplicated rows
duplicated_data = data[data['is_duplicate'] & (data['repeat_count'] == 2)]

# Loop through each group of duplicates
for name, group in duplicated_data.groupby(sort_columns):
    if group.shape[0] == 2:  # only deal with groups that have exactly two observations
        idx1, idx2 = group.index
        aluguel1, aluguel2 = group['aluguel']

        # Compute ratio
        ratio = max(aluguel1, aluguel2) / min(aluguel1, aluguel2)

        # Store ratio in the original DataFrame
        data.loc[idx1, 'aluguel_ratio'] = ratio
        data.loc[idx2, 'aluguel_ratio'] = ratio

        # Classify property types based on ratio
        if ratio >= 30:  # You can choose the ratio based on your domain knowledge
            data.loc[idx1 if aluguel1 > aluguel2 else idx2, 'new_property_type'] = 'Sale'
            data.loc[idx2 if aluguel1 > aluguel2 else idx1, 'new_property_type'] = 'Rent'
        else:
            data.loc[[idx1, idx2], 'new_property_type'] = 'Uncertain'

del aluguel1, aluguel2, duplicated_data, group, name, idx1, idx2, ratio, sort_columns

# Handle rows that are not duplicated or have 'repeat_count' other than 2
data.loc[data['new_property_type'] == '', 'new_property_type'] = 'Uncertain'

data['new_property_type'].value_counts()

# %%% comparando os dois grupos (rent e sale)
print("Summary statistics for Sale:")
print(data[data['new_property_type'] == 'Sale']['aluguel'].describe())

print("\nSummary statistics for Rent:")
print(data[data['new_property_type'] == 'Rent']['aluguel'].describe())

# median
data[data['new_property_type'] == 'Sale']['aluguel'].median()
data[data['new_property_type'] == 'Rent']['aluguel'].median()

data['new_property_type'].value_counts()

# %% latitude e longitude (python)

# =============================================================================
# import pandas as pd
# import numpy as np
# from geopy.geocoders import Nominatim
# 
# # Remove duplicate rows based on the 'CEP' column in the 'data' DataFrame.
# unique_data = data.drop_duplicates(subset=['CEP'])
# 
# # Create a new DataFrame 'cep_df' with unique 'CEP' values and empty 'lat' and 'long' columns
# cep_df = pd.DataFrame(data['CEP'].unique(), columns=['CEP'])
# cep_df['lat'] = np.nan
# cep_df['long'] = np.nan
# 
# # Merge 'cep_df' and 'unique_data' based on the 'CEP' column to add 'Município' and 'Bairro' to 'cep_df'
# cep_df = pd.merge(cep_df, unique_data[['CEP', 'Município', 'Bairro']], on='CEP', how='left')
# del unique_data
# 
# # Initialize geolocator
# geolocator = Nominatim(user_agent="geolocalização")
# 
# # Loop through each CEP to get latitude and longitude
# for idx, row in cep_df.iterrows():
#     try:
#         # Combine CEP, Município, and Bairro into a single search string
#         # search_string = f"{row['CEP']}, {row['Município']}, {row['Bairro']}"
#         # Get location using geolocator
#         # location = geolocator.geocode(search_string)
#         location = geolocator.geocode(row['Bairro'] + ", " + row['Município'] + " - CEP " + str(row['CEP']))
#         # location = geolocator.geocode(str(row['CEP']))
#         if location:
#             cep_df.at[idx, 'lat'] = location.latitude
#             cep_df.at[idx, 'long'] = location.longitude
#     except Exception as e:
#         print(f"An error occurred: {e}")
# 
# # Now cep_df should have 'lat' and 'long' columns filled with latitude and longitude for each unique CEP.
# # import openpyxl
# # cep_df.to_excel('web_scraping\\coord.xlsx', index=False)
# =============================================================================


# %% latitude e longitude (google API no R)

lat_long = pd.read_excel('web_scraping\\long_lat.xlsx')

lat_long.info()

# merge
data = pd.merge(data, lat_long, on="CEP", how="left")



# %% plotando latitude e longitude
import pandas as pd
import matplotlib.pyplot as plt
import descartes
import geopandas as gpd
from shapely.geometry import Point, Polygon

street_map = gpd.read_file('web_scraping\\shapefiles\\RJ_Municipios_2022\\RJ_Municipios_2022.shp')

# =============================================================================
# fig, ax = plt.subplots(figsize=(15, 15))
# street_map.plot(ax=ax)
# =============================================================================

# %% plotando apenas lat e long

# =============================================================================
# plt.figure(figsize=(10, 10))
# plt.scatter(data['long'], data['lat'], c='blue', marker='o')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.title('Latitude and Longitude')
# plt.grid(True)
# plt.show()
# =============================================================================

# removendo outliers
# =============================================================================
# filtered_data = data[data['lat'] >= -25]
# 
# plt.figure(figsize=(10, 10))
# plt.scatter(filtered_data['long'], filtered_data['lat'], c='blue', marker='o')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.title('Filtered Latitude and Longitude')
# plt.grid(True)
# plt.show()
# =============================================================================

# transformando as latitudes e longitudes do dataset em POINT
crs = {'init': 'epsg:4326'}

geometry = [Point(xy) for xy in zip(data['long'], data['lat'])]

# We should end up with a list of Points that we can use to create our GeoDataFrame:
geo_df = gpd.GeoDataFrame(data,  # specify our data
                          crs=crs,  # specify our coordinate reference system
                          geometry=geometry)  # specify the geometry list we created

# Check if there are any empty geometries
print(f"Number of empty geometries: {sum([geom.is_empty for geom in geo_df['geometry']])}")

# removing empty geoms
geo_df = geo_df[geo_df['geometry'].is_empty == False]

geo_df.plot(marker='o', color='blue', markersize=5)
plt.show()

# Remove rows where latitude is less than -25
geo_df = geo_df[geo_df['lat'] >= -25]
geo_df.plot(marker='o', color='blue', markersize=5)
plt.show()

# plotting coords on top of rio de janeiro's map
fig, ax = plt.subplots(figsize=(15, 15))
# First plot the street_map
street_map.plot(ax=ax)
# Now plot the filtered_geo_df on the same axis
geo_df.plot(ax=ax, marker='o', color='red', markersize=5)
# plt.legend(prop={'size': 15})
plt.show()

# %% bairros do municipio do rio de janeiro

# shapefile do municipio do rio de janeiro
#bairros = gpd.read_file('web_scraping\\shapefiles\\Limite_do_Municipio_do_Rio_de_Janeiro\\Limite_do_Municipio_do_Rio_de_Janeiro.shp')

# shapefile do municipio do rio de janeiro por bairros
bairros = gpd.read_file('web_scraping\\shapefiles\\Bairros_CidadeRJ\\Limite_de_Bairros.shp')

# selecionando apenas o municipio
data_mun = data[data['Município'] == 'Rio de Janeiro']

# filtrando por coordenadas nao nulas
data_mun = data_mun[data_mun['lat'].notna()]

# criando variavel no formato Point
geometry = [Point(xy) for xy in zip(data_mun['long'], data_mun['lat'])]

# definindo Coordinate Reference System (CRS)
crs = {'init': 'epsg:4326'}

# We should end up with a list of Points that we can use to create our GeoDataFrame:
geo_df_municipio = gpd.GeoDataFrame(data_mun,
                                    crs=crs,
                                    geometry=geometry)

# colocando shapefile de 'bairros' no mesmo padrao de coordenadas de 'geo_df_municipio'
bairros = bairros.to_crs(geo_df_municipio.crs)

# %% grafico 1

fig, ax = plt.subplots(figsize=(15, 15))
fig.suptitle("Localização dos imóveis anunciados para aluguel ou \nvenda na cidade do Rio de Janeiro", fontsize=30)

# Customizing the bairros plot by changing line width and edge color
bairros.plot(ax=ax,
             color='grey',  # Filling color
             linewidth=1,  # Line thickness
             edgecolor='black'  # Line color
             )

geo_df_municipio.plot(ax=ax, marker='o', color='blue', markersize=5, alpha=0.5)
ax.set_xticks([])
ax.set_yticks([])

# Adding a caption at the bottom
plt.figtext(0.9, 0.1, "Fontes: Elaboração própria a partir de dados da OLX e do Data Rio",
            ha="right",
            fontsize=12
            #bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5}
            )

plt.show()

# %% grafico 1.2
fig, ax = plt.subplots(figsize=(8, 6))
fig.suptitle("Location of properties for rent or for sale in the city of Rio de Janeiro",
             y=0.8, fontsize=12, fontweight='bold')

# Customizing the bairros plot by changing line width and edge color
bairros.plot(ax=ax,
             color='lightgrey',  # Filling color
             linewidth=1,  # Line thickness
             edgecolor='darkgrey'  # Line color
             )

geo_df_municipio.plot(ax=ax, marker='o', color='red', markersize=3, alpha=0.7)
ax.set_xticks([])
ax.set_yticks([])

# Aspect ratio
ax.set_aspect('equal', 'box')

# Adding a caption at the bottom
# =============================================================================
# plt.figtext(0.9, 0.1, "Fonte: Elaboração própria a partir de dados da OLX e do Data Rio",
#             ha="right",
#             fontsize=12
#             #bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5}
#             )
# =============================================================================

plt.savefig('web_scraping\\figuras\\01_mapa_municipio_rj.png', dpi=300, bbox_inches='tight')
plt.show()


# %% grafico 2
# Create a figure with 3 subplots
# =============================================================================
# fig, axs = plt.subplots(1, 3, figsize=(45, 15))
# 
# # Setting a common title for all subplots
# fig.suptitle("Geospatial Data for Different Property Types", fontsize=30)
# 
# # List of property types
# property_types = ['Rent', 'Sale', 'Uncertain']
# 
# # Iterate over each subplot to plot data filtered by property type
# for ax, property_type in zip(axs, property_types):
#     # Plot bairros
#     bairros.plot(ax=ax, color='grey', alpha=0.4)
#     
#     # Filter and plot geo_df_municipio by property type
#     geo_df_filtered = geo_df_municipio[geo_df_municipio['new_property_type'] == property_type]
#     geo_df_filtered.plot(ax=ax, marker='o', color='blue', markersize=5)
#     
#     # Customization
#     ax.set_title(f"{property_type} Properties", fontsize=20)
#     ax.set_xticks([])
#     ax.set_yticks([])
# 
# plt.show()
# =============================================================================


# %% write csv to make a random forest prediction in another script
#data_mun.to_csv('web_scraping\\data_mun_rf.csv', index=False, encoding='latin-1')
