import dgl
import torch as th
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch.utils.data as Data

# 边 0->1, 0->2, 0->3, 1->3
node = np.load('./nodenumber.npy',allow_pickle=True)
nodeattr = np.load('./nodeattribute.npy',allow_pickle=True)
edges = np.load('./edges.npy',allow_pickle=True)

print(nodeattr.shape)
#
# def relu(x):
#     return (abs(x) + x) / 2
#
# (m,) = edges.shape
# a = list()
# GCNdrug_feature = np.zeros((m,64))
# for i in range(m):
#     edge = edges[i]
#     u = edge[:,0]
#     v = edge[:,1]
#     g = dgl.graph((u, v))               ##构造分子图
#     G = nx.Graph()
#     G.add_nodes_from(range(0,len(g.nodes())))
#     G.add_edges_from(edge)
#     A=nx.adjacency_matrix(G).todense()            ##邻接矩阵
#     p = len(g.nodes())
#     # # 打印图
#     # fig, ax = plt.subplots()
#     # nx.draw(g.to_networkx(), ax=ax, with_labels=True)
#     # plt.show()
#     ##输入一些值
#     # print(g)  # 图中节点的数量是DGL通过给定的图的边列表中最大的点ID推断所得出的
#     # # 获取节点的ID
#     # print(g.nodes())
#     # 获取边的对应端点
#     # print(g.edges())
#     # # 获取边的对应端点和边ID
#     # print(g.edges(form='all'))
#     # # 如果具有最大ID的节点没有边，在创建图的时候，用户需要明确地指明节点的数量。
#     # # g = dgl.graph((u, v), num_nodes=8)
#
#     g.ndata['feats'] = th.from_numpy(nodeattr[i])     ##给节点属性赋值
#     # print(g.node_attr_schemes()) #输出图中每一个特征shape等信息
#     X = g.ndata['feats']              ##节点的属性
#
#     I = np.eye(p, dtype=int)
#     A_hat = A + I  ##自环的度矩阵
#     D_hat = np.array(np.sum(A_hat, axis=0))[0]
#     D_hat = np.matrix(np.diag(D_hat))
#
#     W_1 = np.random.normal(
#         loc=0, scale=1, size=(g.number_of_nodes(), 128))
#
#     W_2 = np.random.normal(
#         loc=0, size=(W_1.shape[1], 64))
#
#
#     def gcn_layer(A_hat, D_hat, X, W):
#         return relu(D_hat ** -1 * A_hat * X * W)
#
#
#     H_1 = gcn_layer(A_hat, D_hat, I, W_1)
#     H_2 = gcn_layer(A_hat, D_hat, H_1, W_2)
#     Max_Pool = np.max(H_2, axis=0)
#     Mean_Pool = np.average(H_2, axis=0)
#     output= Max_Pool
#
#     # feature_representations = {
#     #     node: np.array(output)[node]
#     #     for node in g.nodes()}
#     #
#     # print(feature_representations)
#
#     GCNdrug_feature[i,:]=output
#
# print(GCNdrug_feature.shape)
# np.save('GCNdrug_feature.npy', GCNdrug_feature)
