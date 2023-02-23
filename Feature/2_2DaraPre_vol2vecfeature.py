####将GCNdrug_feature.npy中得到GCN提取特征后的药物特征表示，和Input.csv中对应起来，形成drug_gcn.npy

import numpy as np
import torch
import csv

gcn_feature = np.load('mol2vec_feature.npy')
[m,n] = gcn_feature.shape
drug=list()
with open('imput_nr.csv', 'r') as csvfile:               ##############
    read = csv.reader(csvfile)
    for index, value in enumerate(read):
        if (index != 0):
            drug.append(value[0])

drug_nr=list()
with open('nr.csv', 'r') as csvfile:               ###########
    read = csv.reader(csvfile)
    for index, value in enumerate(read):
        drug_nr.append(value[1])

m = len(drug_nr)                   #m是90
drug_gcn = np.zeros((m,n))
for i in range(m):
    index = drug.index(drug_nr[i])                 #drug_nr中药物在drug中的位置
    drug_gcn[i,:]= gcn_feature[index]

np.save('drug_vol2vec_nr.npy',drug_gcn)




