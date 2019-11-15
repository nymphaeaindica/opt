import math
import pandas as pd
from sklearn.metrics import roc_curve
import csv
import plotly.graph_objs as go
 
 
def sigmoid(x):
  return 1 / (1 + math.exp(-x))
 
def converter(str):
    list = str.split(',')
    list = [float(i) for i in list]
    return list
 
labels = pd.read_csv('labels.csv', delimiter=',', header=None)
labels = labels.drop([0], axis=1)
predictions = []
with open('predictions.csv') as f:
    data = csv.reader(f)
    for row in data:
        row = [float(i) for i in row]
        predictions.append(row)
 
def opt(list_of_lists, thr):
    updated = []
    for list in list_of_lists:
        sum = 0
        # thr = 0.4
        for item in list:
           if item > thr:
               x = (item - thr) / (1 - thr)
               sum = sum + x
        upd = sigmoid(sum)
        updated.append(upd)
    return updated
 
fig = go.Figure()
threshold = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
for thr in threshold:
    scores = opt(predictions, thr)
    fpr, tpr, t = roc_curve(labels, scores)
    fig.add_trace(go.Scatter(x=fpr, y=tpr, name=thr))
fig.show()
 
 
