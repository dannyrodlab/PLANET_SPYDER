# -*- coding: utf-8 -*-
"""
Created on Sun May 19 18:19:20 2019

@author: BATMAN
"""
import argparse
parser = argparse.ArgumentParser('TELL ME WHAT TO DO!')
parser.add_argument('--file', type=str, required=True)
args = parser.parse_args()
PLANET_KAGGLE_DATA_NAME = args.file

## Set folders
import os
PLANET_KAGGLE_ROOT = os.path.abspath("./results/")
print(PLANET_KAGGLE_ROOT)
assert os.path.exists(PLANET_KAGGLE_ROOT)
PLANET_KAGGLE_FIGURES = os.path.abspath("./figures/")
print(PLANET_KAGGLE_FIGURES)
assert os.path.exists(PLANET_KAGGLE_FIGURES)
PLANET_KAGGLE_LABEL_CSV = os.path.join(PLANET_KAGGLE_ROOT, PLANET_KAGGLE_DATA_NAME + '.txt' )
assert os.path.exists(PLANET_KAGGLE_LABEL_CSV)

## np.savetxt(name, (history_acc, history_loss, history_val_acc, history_val_loss), delimiter=',')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
## TODO: Integer Axis --> from matplotlib.ticker import MaxNLocator

## Read some data motherfucker
df_labels = pd.read_csv(PLANET_KAGGLE_LABEL_CSV, sep=",", header=None)
df_labels = df_labels.T
df_labels.columns = ["acc", "loss", "val_acc", "val_loss"]
print(df_labels.head())

acc = df_labels.loc[: , "acc"]
loss = df_labels.loc[: , "loss"]
val_acc = df_labels.loc[: , "val_acc"]
val_loss = df_labels.loc[: , "val_loss"]

number_of_epochs = df_labels['acc'].shape[0]

# evenly sampled time at 200ms intervals
epochs = np.arange(1., number_of_epochs + 1.0, 1.0)

plt.subplot(2, 1, 1)
plt.plot(epochs, acc,'--bo', label='acc')
plt.plot(epochs, loss,'--ro', label='loss') 
plt.legend(loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('Value []')
plt.ylim((0,1.50))
plt.title(PLANET_KAGGLE_DATA_NAME)

plt.subplot(2, 1, 2)
plt.plot(epochs, val_acc,'--bo', label='val_acc')
plt.plot(epochs, val_loss,'--ro', label='val_loss') 
plt.legend(loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('Value []')
plt.ylim((0,1.50))
plt.title(PLANET_KAGGLE_DATA_NAME)

plt.subplots_adjust(hspace=0.75,
                    wspace=0.35)

plt.savefig('figures/' + PLANET_KAGGLE_DATA_NAME + '.png', dpi=100)

plt.show()


