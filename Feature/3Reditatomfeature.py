import numpy as np
import torch
import dgl
from dgl import DGLGraph
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdDesc
import csv

#
# # molecule_smiles='[C@@H](Cl)(F)Br'
# molecule_smiles= 'CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5.CS(=O)(=O)O'
# #加载smile生成mol对象
# molecule = Chem.MolFromSmiles(molecule_smiles)

#
mol_text=list()
with open('imput_gpcr.csv', 'r') as csvfile:
    read = csv.reader(csvfile)
    for index, value in enumerate(read):
        if (index != 0):
            mol_text.append(value[3])              #保存所有的molecule_smiles
m = len(mol_text)

#辅助函数
def one_of_k_encoding_unk(x, allowable_set):
    '将x与allowable_set逐个比较，相同为True， 不同为False, 都不同则认为是最后一个相同'
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))
##元素种类、隐含价、价电子、成键、电荷、杂化类型
def get_atom_features(atom):
    possible_atom = ['C', 'N', 'O', 'F', 'P', 'Cl', 'Br', 'I', 'DU'] #DU代表其他原子
    atom_features = one_of_k_encoding_unk(atom.GetSymbol(), possible_atom)
    atom_features += one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1])
    atom_features += one_of_k_encoding_unk(atom.GetNumRadicalElectrons(), [0, 1])
    atom_features += one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6])
    atom_features += one_of_k_encoding_unk(atom.GetFormalCharge(), [-1, 1])
    atom_features += one_of_k_encoding_unk(atom.GetHybridization(),
                                           [Chem.rdchem.HybridizationType.SP,
                                            Chem.rdchem.HybridizationType.SP2,
                                            Chem.rdchem.HybridizationType.SP3,
                                            Chem.rdchem.HybridizationType.SP3D])
    return np.array(atom_features)
#单键、双键、三键、成环、芳香环、共轭
def get_bond_features(bond):
    bond_type = bond.GetBondType()
    bond_feats = [
        bond_type == Chem.rdchem.BondType.SINGLE, bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE, bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    return np.array(bond_feats)

def features(molecule):
    G.add_nodes(molecule.GetNumAtoms())
    node_features = []
    edge_features = []

    for i in range(molecule.GetNumAtoms()):
        atom_i = molecule.GetAtomWithIdx(i)
        atom_i_features = get_atom_features(atom_i)
        node_features.append(atom_i_features)

        for j in range(molecule.GetNumAtoms()):
            bond_ij = molecule.GetBondBetweenAtoms(i, j)
            if bond_ij is not None:
                G.add_edges(i, j)
                bond_features_ij = get_bond_features(bond_ij)
                edge_features.append(bond_features_ij)

    G.ndata['x'] = torch.from_numpy(np.array(node_features))  # dgl添加原子/节点特征
    G.edata['w'] = torch.from_numpy(np.array(edge_features))  # dgl添加键/边特征

    return G.ndata['x'], G.edata['w']

attrlist=list()
attr = np.empty(m, object)             ##保存分子节点的attribute保存不同大小的object
edge = np.empty(m, object)
node = np.empty(m, object)
for i in range(m):
    G = DGLGraph()
    molecule = Chem.MolFromSmiles(mol_text[i])               ##批量读取SMILES格式
    a, b = features(molecule)                                ##利用feature给每个原子添加特征
    attrlist.append((a.int()).numpy())
attr[:] = attrlist

np.save('nodeattribute.npy',attr)


from pysmiles import read_smiles
import networkx as nx
import matplotlib.pyplot as plt
edgelist=list()
nodelist = list()
list1 = list()
list2 = list()
for i in range(1):
    mol = read_smiles(mol_text[i]) # 读取到一个networkx的网络结构中
    # print(mol.nodes) # 打印节点信息
    # print(mol.edges) # 打印边信息
    ###保存边信息
    a = np.array(mol.edges)
    edgelist.append(np.array(mol.edges))
    ###保存点编号信息
    elements = nx.get_node_attributes(mol, name="element")
    nodelist.append(elements)


edge[:] = edgelist
node[:] = nodelist
# np.save('edges.npy',edge)
# np.save('nodenumber.npy',node)





