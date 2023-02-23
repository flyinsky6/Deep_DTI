##graph2vec的数据准备，生成包含edges和度特征的jason数据，保存在graph2vec目录中。
import numpy as np
import torch
import dgl
from dgl import DGLGraph
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdDesc
import csv
import json


#
mol_text=list()
with open('imput_nr.csv', 'r') as csvfile:
    read = csv.reader(csvfile)
    for index, value in enumerate(read):
        if (index != 0):
            mol_text.append(value[3])              #保存所有的molecule_smiles
m = len(mol_text)


attrlist=list()
attr = np.empty(m, object)             ##保存分子节点的attribute保存不同大小的object
edge = np.empty(m, object)
node = np.empty(m, object)


from pysmiles import read_smiles
import networkx as nx
import matplotlib.pyplot as plt
edgelist=list()
nodelist = list()

degreefeature = dict()
for i in range(m):
    list1 = list()
    list2 = list()
    mol = read_smiles(mol_text[i]) # 读取到一个networkx的网络结构中
    # print(mol.nodes) # 打印节点信息
    # print(mol.edges) # 打印边信息
    ###保存边信息
    a = np.array(mol.edges)
    edgelist.append(np.array(mol.edges))
    ###保存点编号信息
    elements = nx.get_node_attributes(mol, name="element")
    nodelist.append(elements)
    temp = mol.degree
    k = len(nodelist[i])

    for j in range(k):
        list1.append(str(temp[j]))
        list2.append(str(j))
    degreefeature = dict(zip(list2, list1))
    temp = {'edges': (edgelist[i]).tolist(), 'features':degreefeature}
    j = json.dumps(temp)
    with open('./graph2vecdataset/nr/'+ str(i) +'.json', 'w') as json_file:
        json_file.write(j)
    json_file.close()






