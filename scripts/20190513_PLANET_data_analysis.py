## Set folders
import os
PLANET_KAGGLE_ROOT = os.path.abspath("../input/")
assert os.path.exists(PLANET_KAGGLE_ROOT)
PLANET_KAGGLE_FIGURES = os.path.abspath("figures/")
assert os.path.exists(PLANET_KAGGLE_FIGURES)
PLANET_KAGGLE_LABEL_CSV = os.path.join(PLANET_KAGGLE_ROOT, 'train_v2.csv')
assert os.path.exists(PLANET_KAGGLE_LABEL_CSV)

## Plot parameters
figures_size = (8,8)
figures_ext = '.png'

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

##
df_labels = pd.read_csv(PLANET_KAGGLE_LABEL_CSV)
print(df_labels.head())

# Build list with unique labels
label_list = []
for tag_str in df_labels.tags.values:
    labels = tag_str.split(' ')
    for label in labels:
        if label not in label_list:
            label_list.append(label)
            
print(label_list)

# Add onehot features for every label
for label in label_list:
    df_labels[label] = df_labels['tags'].apply(lambda x: 1 if label in x.split(' ') else 0)
# Display head
print(df_labels.head())

# Histogram of label instances
print(df_labels[label_list].sum().sort_values())
plt.figure(figsize=figures_size)
df_labels[label_list].sum().sort_values().plot.bar()
figure_name = '20190513_PLANET_DISTRIBUTION_LABELS.png'
plt.savefig(os.path.join(PLANET_KAGGLE_FIGURES, figure_name), format='png', dpi=1000)


def make_cooccurence_matrix(labels):
    numeric_df = df_labels[labels]; 
    c_matrix = numeric_df.T.dot(numeric_df)
    return c_matrix
    
# Compute the co-ocurrence matrix


c_matrix_all = make_cooccurence_matrix(label_list)
plt.figure(figsize=(16,16))
sns.heatmap(c_matrix_all, annot=True, fmt="d", linewidths=0, cmap="YlGnBu", square=True)
figure_name = '20190513_PLANET_COOCURRENCE_ALL.png'
plt.savefig(os.path.join(PLANET_KAGGLE_FIGURES, figure_name), format='png', dpi=1000)

weather_labels = ['clear', 'partly_cloudy', 'haze', 'cloudy']
c_matrix_weather = make_cooccurence_matrix(weather_labels)
plt.figure(figsize=figures_size)
sns.heatmap(c_matrix_weather, annot=True, fmt="d", linewidths=0, cmap="YlGnBu", square=True)
figure_name = '20190513_PLANET_COOCURRENCE_WEATHER.png'
plt.savefig(os.path.join(PLANET_KAGGLE_FIGURES, figure_name), format='png', dpi=1000)

land_labels = ['primary', 'agriculture', 'water', 'cultivation', 'habitation', 'road']
c_matrix_land = make_cooccurence_matrix(land_labels)
plt.figure(figsize=figures_size)
sns.heatmap(c_matrix_land, annot=True, fmt="d", linewidths=0, cmap="YlGnBu", square=True)
figure_name = '20190513_PLANET_COOCURRENCE_LAND.png'
plt.savefig(os.path.join(PLANET_KAGGLE_FIGURES, figure_name), format='png', dpi=1000)

rare_labels = [l for l in label_list if df_labels[label_list].sum()[l] < 2000]
c_matrix_rare = make_cooccurence_matrix(rare_labels)
plt.figure(figsize=figures_size)
sns.heatmap(c_matrix_rare, annot=True, fmt="d", linewidths=0, cmap="YlGnBu", square=True)
figure_name = '20190513_PLANET_COOCURRENCE_RARE.png'
plt.savefig(os.path.join(PLANET_KAGGLE_FIGURES, figure_name), format='png', dpi=1000)