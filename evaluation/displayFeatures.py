#!/usr/bin/env python3

import argparse
import glob
import pandas as pd
import os
import torch
import librosa
import random
import seaborn as sns
import numpy as np

from sklearn.manifold import TSNE

from fine_tune.trainer import BEATsTransferLearningModel

from BEATs.Tokenizers import TokenizersConfig, Tokenizers
from BEATs.BEATs import BEATs, BEATsConfig
import json
import pickle
import matplotlib.pyplot as plt

# this is to display clustered data based on extracted BEATs features, purely for testing purposes

def loadBEATs():
    # load the pre-trained checkpoints
    finetuned_1 = torch.load('/home/ubuntu/shared-dir/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt')

    cfg = BEATsConfig(finetuned_1['cfg'])
    BEATs_model = BEATs(cfg)
    BEATs_model.load_state_dict(finetuned_1['model'])
    BEATs_model.eval()
    return BEATs_model

def extractFeatures(BEATs_model, trs):
    for t in trs:
        print(t)
        padding_mask = torch.zeros(t.shape[0], t.shape[1]).bool()
        representation = BEATs_model.extract_features(t, padding_mask=padding_mask)[0]
        if representation.dim() == 3:
            representation = representation[:, -1, :]
        yield representation.detach().numpy()

def get_2d_features(features, perplexity):
    representation = np.concatenate(np.array(features), axis=0)
    tsne = TSNE(n_components=2, perplexity=perplexity)
    features_2d = tsne.fit_transform(representation)

    return features_2d

def get_figure(features_2d, labels, fig_name):
    fig = sns.scatterplot(x=features_2d[:, 0], y=features_2d[:, 1], hue=labels)
    sns.move_legend(fig, "upper left", bbox_to_anchor=(1, 1))
    fig.get_figure().savefig(fig_name, bbox_inches="tight")

def flatten_labels(features_2d, labels):
    flattened_labels = []
    flattened_features = []
    for i in range(len(labels)):
        row = labels[i]
        non_none_labels = [label for label in row if label is not None]
        if len(non_none_labels) == 1:
            flattened_labels.append(non_none_labels[0])
            flattened_features.append(features_2d[i])
    return flattened_labels, flattened_features

trs, labels = [], []
print("[INFO] Loading the BEATs model")
BEATs_model = loadBEATs()

print("[INFO] Getting the features")
features = list(extractFeatures(BEATs_model, trs))

print("[INFO] Reducing the dimensions of the features to 2D")
features_2d = get_2d_features(features, perplexity=5)


print(len(labels), labels.shape)
print(labels)
flattened_labels, flattened_features = flatten_labels(features_2d, labels)

# Print the results for debugging
print(len(flattened_features))
print(len(flattened_labels))
print(flattened_labels)

fig_name = 'tsne.png'

def get_figure_new(features_2d, labels, fig_name):
    fig, ax = plt.subplots(figsize=(10, 8))

    # Flatten the labels and features_2d for plotting
    flattened_features = []
    flattened_labels = []

    for i in range(len(labels)):
        for label in labels[i]:
            if label is not None:
                flattened_features.append(features_2d[i])
                flattened_labels.append(label)

    flattened_features = np.array(flattened_features)
    flattened_labels = np.array(flattened_labels)

    # Use a unique set of labels for coloring
    unique_labels = np.unique(flattened_labels)
    label_to_color = {label: sns.color_palette('tab20', n_colors=len(unique_labels))[i]
                      for i, label in enumerate(unique_labels)}

    # Plot each label separately to preserve multi-label nature
    for label in unique_labels:
        idx = flattened_labels == label
        ax.scatter(flattened_features[idx, 0], flattened_features[idx, 1], color=label_to_color[label], label=label)

    # Add legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Save the figure
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close(fig)  # Close the figure to free up memory
    
print("[INFO] Making the figure")
# flattened_features = np.array(flattened_features)
# flattened_labels = np.array(flattened_labels)
fig_name = 'tsne.png'
get_figure_new(features_2d, labels, fig_name)
files.download(fig_name)
