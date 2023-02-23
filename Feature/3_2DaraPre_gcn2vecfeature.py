####将GCNdrug_feature.npy中得到GCN提取特征后的药物特征表示，和Input.csv中对应起来，形成drug_gcn.npy

import numpy as np
import torch
import csv
import pandas as pd


data = pd.read_csv("./graph2vecfeatures/nr.csv",index_col=1)
np.save("graph2vecfeatures_nr.npy", data)

#
gcn_feature = np.load('graph2vecfeatures_ic.npy',allow_pickle=True)
[m,n] = gcn_feature.shape
drug=list()
with open('imput_nr.csv', 'r') as csvfile:
    read = csv.reader(csvfile)
    for index, value in enumerate(read):
        if (index != 0):
            drug.append(value[0])

drug_nr=list()
with open('nr.csv', 'r') as csvfile:
    read = csv.reader(csvfile)
    for index, value in enumerate(read):
        drug_nr.append(value[1])

m = len(drug_nr)                   #m是90
drug_gcn = np.zeros((m,n))
for i in range(m):
    index = drug.index(drug_nr[i])                 #drug_nr中药物在drug中的位置
    drug_gcn[i,:]= gcn_feature[index]

print(drug_gcn.shape)
np.save('drug_graph2vec_nr.npy',drug_gcn)




