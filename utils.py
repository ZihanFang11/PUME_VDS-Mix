
import torch
import h5py
import scipy.io as sio
import scipy.sparse as ss
from sklearn.preprocessing import normalize
def distance(X, Y, square=True):
    """
    计算两组样本之间的欧几里得距离。

    参数:
        X (torch.Tensor): 样本集合，维度为d*n。
        Y (torch.Tensor): 样本集合，维度为d*m。
        square (bool): 是否返回距离的平方。

    返回:
        torch.Tensor: 距离矩阵，维度为n*m。
    """
    n = X.shape[1]
    m = Y.shape[1]
    x = torch.norm(X, dim=0)
    x = x * x
    x = torch.t(x.repeat(m, 1))

    y = torch.norm(Y, dim=0)
    y = y * y
    y = y.repeat(n, 1)
    crossing_term = torch.t(X).matmul(Y)
    result = x + y - 2 * crossing_term
    result = result.relu()
    if not square:
        result = torch.sqrt(result)
    return result
def build_CAN(X, num_neighbors, links=0):
    """
    Solve Problem: Clustering-with-Adaptive-Neighbors(CAN)
    :param X: d * n
    :param num_neighbors:
    :return: Graph
    """
    size = X.shape[1]
    num_neighbors = min(num_neighbors, size - 1)
    distances = distance(X, X)
    distances = torch.max(distances, torch.t(distances))
    sorted_distances, _ = distances.sort(dim=1)
    top_k = sorted_distances[:, num_neighbors]
    top_k = torch.t(top_k.repeat(size, 1)) + 10 ** -10

    sum_top_k = torch.sum(sorted_distances[:, 0:num_neighbors], dim=1)
    sum_top_k = torch.t(sum_top_k.repeat(size, 1))
    torch.cuda.empty_cache()
    T = top_k - distances
    torch.cuda.empty_cache()
    weights = torch.div(T, num_neighbors * top_k - sum_top_k)
    torch.cuda.empty_cache()
    weights = weights.relu().cpu()
    if links != 0:
        links = torch.Tensor(links).to(X.device)
        weights += torch.eye(size).to(X.device)
        weights += links
        weights /= weights.sum(dim=1).reshape([size, 1])
    torch.cuda.empty_cache()
    raw_weights = weights
    weights = (weights + weights.t()) / 2
    raw_weights = raw_weights.to(X.device)
    weights = weights.to(X.device)
    return weights, raw_weights

def build_graphs(X, neighbor):
    L = []
    # 遍历每个视图的数据
    for v in range(len(X)):
        X_v = torch.tensor(X[v]).T  # 转置数据
        A_v, _ = build_CAN(X_v, neighbor)  # 使用工具函数构建邻接矩阵
        L.append(torch.eye(A_v.shape[0])-A_v)  # 添加到列表
    return L




def load_data(dataset, path="./data/"):
    feature_list = []
    if dataset == "AwA":
        data = h5py.File(path + dataset + '.mat')
        features = data['X']
        for i in range(features.shape[1]):
                feature_list.append(normalize(data[features[0][i]][:].transpose()))
        labels = data['Y'][:].flatten()
    else:
        data = sio.loadmat(path + dataset + '.mat')
        features = data['X']
        for i in range(features.shape[1]):
            features[0][i] = normalize(features[0][i])
            feature = features[0][i]
            if ss.isspmatrix_csr(feature):
                feature = feature.todense()
                print("sparse")
            feature_list.append(feature)
        labels = data['Y'].flatten()
    labels = labels - min(set(labels))
    return feature_list, labels