import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import TensorDataset, DataLoader, random_split, ConcatDataset
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch

# function for the model
class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.clf = nn.Sequential(
            nn.Linear(self.in_features, self.out_features))

    # Making the model callable
    def forward(self, inputs):
        outputs = self.clf(inputs)
        return outputs

    # prediction with the probabilities greater than 0.5
    def predict_single(self, input, labels):
        outputs = self(input)
        probs = torch.sigmoid(outputs)
        probs = [(idx, labels[idx], probs[idx].detach().item()) for idx in range(
            len(probs)) if probs[idx] >= 0.5 or torch.max(probs) == probs[idx]]
        return probs

# function to test the input symptoms
def verify(symptom_one, symptom_two, symptom_three, symptom_four, symptom_five):
    label_csv = 'label.csv'
    label_df = pd.read_csv(label_csv)
    labels = dict(
        enumerate(label_df['prognosis'].astype('category').cat.categories))
    classes = label_df['prognosis'].describe().loc['unique']
    in_features = int(43)

    # loading the model file to pass the values
    model = Model(in_features,classes)
    m = torch.load('model/model.pth')

    # from the fourty three symptoms, making true for three symptoms from user input
    i = torch.zeros(43)
    i[int(symptom_one)] = 1
    i[int(symptom_two)] = 1
    i[int(symptom_three)] = 1
    i[int(symptom_four)] = 1
    i[int(symptom_five)] = 1

    # input values to the model
    input, output = (i, 1)

    # return predicted value
    return model.predict_single(input, labels)
