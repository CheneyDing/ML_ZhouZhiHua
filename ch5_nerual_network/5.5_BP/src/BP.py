import numpy as np
import pandas as pd


def loadDataSet(file_name, data_file_encode):
    f = open(file_name, mode='r', encoding=data_file_encode)
    df = pd.read_csv(f)
    wm_df = pd.get_dummies(df)
    x = wm_df[wm_df.columns[1:-2]]
    y = wm_df[wm_df.columns[-2:]]
    label = wm_df.columns.data[-2:]
    # print(x); print(y); print(label)
    return x, y, label


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_layer_output(x, v, bias_v, w, bias_w):
    a = []; b = []
    for i in range(h_num):
        a.append(sigmoid(np.dot(x, v[:, i]) - bias_v[i]))
    for i in range(o_num):
        b.append(sigmoid(np.dot(a, w[:, i]) - bias_w[i]))
    return np.array(a), np.array(b)


def compute_layer_gradient(y, y_l, b_h, w):
    g = []; e = []
    for j in range(o_num):
        g.append(y_l[j] * (1.0 - y_l[j]) * (y[j] - y_l[j]))
    for h in range(h_num):
        e.append(b_h[h] * (1 - b_h[h]) * (np.dot(w[h], g)))
    return np.array(g), np.array(e)


if __name__ == "__main__":
    data_file_encode = "gb18030"
    x, y, label = loadDataSet("../data/watermelon_3.csv", data_file_encode)
    s_num = x.shape[0]; i_num = x.shape[1]; o_num = y.shape[1]; h_num = 5
    v = np.random.rand(i_num, h_num)
    bias_v = np.random.rand(h_num)
    w = np.random.rand(h_num, o_num)
    bias_w = np.random.rand(o_num)
    # print(v); print(w); print(bias_v); print(bias_w)
    step = 0.01
    train_times = 1000
    # train
    for t in range(train_times):
        # for each sample
        for n in range(s_num):
            b_h, y_l = compute_layer_output(x.iloc[n], v, bias_v, w, bias_w)
            g, e = compute_layer_gradient(y.iloc[n], y_l, b_h, w)
            for h in range(h_num):
                for j in range(o_num):
                    w[h][j] += step * g[j] * b_h[h]
            for j in range(o_num):
                bias_w[j] -= step * g[j]
            for i in range(i_num):
                for h in range(h_num):
                    v[i][h] += step * e[h] * x.iloc[n][i]
            for h in range(h_num):
                bias_v[h] -= step * e[h]
        # predict
        _, y_l0 = compute_layer_output(x.iloc[0], v, bias_v, w, bias_w)
        print(y_l0)
        # compute loss
        loss = 0
        for j in range(o_num):
            loss += 0.5 * np.square(y.iloc[0][j] - y_l0[j])
        print('train_times: ' + str(t) + ' loss: ' + str(loss))
    # test error
    error = 0
    for i in range(s_num):
        _, y_pre = compute_layer_output(x.iloc[i], v, bias_v, w, bias_w)
        if (y_pre[0] - y_pre[1]) * (y.iloc[i][0] - y.iloc[i][1]) < 0:
            error += 1
    print('error rate: ' + str(error / s_num))





