import torch


# 网络定义
def get_net():
    return


# 模型训练
def train():
    train_ls, test_ls = 0, 0
    # for epoch in range(:
    # ……………………
    # ……………………
    # 返回训练误差和测试误差
    return train_ls, test_ls


# K折交叉验证
def get_k_fold_data(k, i, X, y):
    # 返回第 i 折交叉验证时所需的训练和验证数据
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_vaild, y_vaild = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)
    return X_train, y_train, X_vaild, y_vaild


# 在 K 折交叉验证中训练 K 次并返回训练和验证的平均误差
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs)
        train_l_sum += train_ls
        valid_l_sum += valid_ls
    return train_l_sum / k, valid_l_sum / k


