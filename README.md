# Deep_DTI
Deep_DTI is a deep learning based method used for the prediction of drug-targeted interaction. Firstly, the CNN network is constructed to represent the potential features of the target's location specific scoring matrix (PSSM); Then, based on the SMILES of drugs as the input, the molecular diagram of drugs is constructed through the open source software RDKit and the atomic features are extracted, and the structural data of the diagram is input into the GCN layer to learn the potential patterns in the drug map feature representation; Finally, the processed eigenvectors are input into the fully connected neural network to predict the drug-target interaction. 
# Requirement
Backend = Tensorflow(1.14.0)
keras(2.3.1)
Numpy(1.20.2)
scikit-learn(1.0.2)
pandas(1.3.5)
matplotlib(3.5.2)
pubchempy(1.0.4)
rdkit(2022.03.2)
dgl(0.6.1)
# Dataset
The data set contains four different sub-datasets including enzymes, ion channels, GPCRs, and nuclear receptors. The information of drug-target interactions can be downloaded from the KEGG BRITE, BRENDA, Super Target, and Drug Bank databases. There are 445,210,223 and 54 drug compounds, which interact with 664, 204, 95, and 26 target proteins, respectively. The number of known interactions is 2926, 1476, 635, and 90, respectively. Then, all known interactions of the drug-target pairs are chosen as positive sample sets in our experiment.And then then select the same number of data as the positive sample from the uncorrelated interaction data as the negative sample.
# Contact
Feel free to contact us if you nedd any help: flyinsky6@gmail.com

