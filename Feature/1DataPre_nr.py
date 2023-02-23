###

import numpy as np
import scipy.io as scio
import pandas as pd
data_path="bind_orfhsa_drug_gpcr.mat"
#Method 1
listhsa=list()
listD=list()
data = scio.loadmat(data_path)
print(data.keys())
# data=data.get('bind_orfhsa_drug_nr')#取出字典里的data
data=data['bind_orfhsa_drug_gpcr']#取出字典里的data
[m,n] = data.shape
print(data.shape)
for i in range(m):
    temp = data[i][0]
    listhsa.append(temp[0])
    temp = data[i][1]
    listD.append(temp[0])

hsa = np.array(listhsa)
D = np.array(listD)

print(hsa.shape)
print(D.shape)

hsa_D = (np.vstack((hsa,D))).transpose()
print(hsa_D.shape)
#
np.save('gpcr.npy',hsa_D)
#
data = pd.DataFrame(hsa_D)
data.to_csv('gpcr.csv',header = False, index = False)

