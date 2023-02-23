###https://www.freesion.com/article/4297543277/

#导入依赖库
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
import warnings
warnings.filterwarnings("ignore")

#载入数据，并查看
data = pd.read_csv('imput_e.csv')
data.head()


#SMILES 转 MOL
data['mol'] = data['smiles'].apply(lambda x: Chem.MolFromSmiles(x))
#计算分子描述符
data['tpsa'] = data['mol'].apply(lambda x: Descriptors.TPSA(x))
data['mol_w'] = data['mol'].apply(lambda x: Descriptors.ExactMolWt(x))
data['num_valence_electrons'] = data['mol'].apply(lambda x: Descriptors.NumValenceElectrons(x))
data['num_heteroatoms'] = data['mol'].apply(lambda x: Descriptors.NumHeteroatoms(x))

# print(data['num_valence_electrons'])

#加载预先经过word2vec训练的模型
from gensim.models import word2vec
model = word2vec.Word2Vec.load('model_300dim.pkl')

#导入mol2vec，计算分子矢量特征，训练模型且结果绘图
from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec
from gensim.models import word2vec
#Constructing sentences
data['sentence'] = data.apply(lambda x: MolSentence(mol2alt_sentence(x['mol'], 1)), axis=1)
#Extracting embeddings to a numpy.array
#Note that we always should mark unseen='UNK' in sentence2vec() so that model is taught how to handle unknown substructures
data['mol2vec'] = [DfVec(x) for x in sentences2vec(data['sentence'], model, unseen='UNK')]

mol2vec_feature = np.array([x.vec for x in data['mol2vec']])
# X_mol = pd.DataFrame(X_mol)

print(mol2vec_feature.shape)

np.save('mol2vec_feature.npy', mol2vec_feature)


