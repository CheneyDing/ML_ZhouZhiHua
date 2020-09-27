import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import pandas as pd
import time


# definition of watermelon dataset
class WaterMelonDataSet(Dataset):
    def __init__(self, csv_file, file_encode):
        df = pd.read_csv(csv_file, encoding=file_encode)
        self.df = pd.get_dummies(df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx, 1:].values


# definition of network structure
class Net(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_output):
        '''
        @parameter n_features: Number of features
        @parameter n_hidden: Number of hidden layer neurons
        @parameter n_output: Number of output layer neurons
        '''
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_features, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        hidden_output = F.relu(self.hidden(x))
        predict_output = self.predict(hidden_output)
        return predict_output


def ComputeAcc(net ,dl):
    correct = 0
    total = 0
    for i, data in enumerate(dl):
        out = net(data[:, 0:-2].float())
        for j in range(len(out)):
            correct += 1 if (out[j][0] > out[j][1] and data[j, -2:][0] > data[j, -2:][1]) \
                            or (out[j][0] < out[j][1] and data[j, -2:][0] < data[j, -2:][1]) else 0
        total += len(out)
    return correct / total


if __name__ == "__main__":
    # load dataset
    ds_watermelon = WaterMelonDataSet("../data/watermelon_3.csv", file_encode="gb18030")
    dl = torch.utils.data.DataLoader(ds_watermelon, batch_size=20, shuffle=False, num_workers=0)

    # build network
    net = Net(n_features=19, n_hidden=5, n_output=2)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    loss_func = torch.nn.MSELoss()

    # train
    train_times = 500
    start_time = time.time()
    for t in range(train_times):
        # train by batch size
        for i, data in enumerate(dl):
            out = net(data[:, 0:-2].float())
            loss = loss_func(out, data[:, -2:].float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # compute accuracy
        acc = ComputeAcc(net, dl)
        print('times: ' + str(t) + ' loss: ' + str(loss) + ' acc: ' + str(acc))
    end_time = time.time()
    print('train time: ' + str(end_time - start_time))