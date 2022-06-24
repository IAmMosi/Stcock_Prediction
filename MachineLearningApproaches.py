import numpy as np  
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import matplotlib.pylab as plt
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.metrics import classification_report
from sklearn import tree
import graphviz
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn 
import matplotlib.pyplot as plt
import statistics



class RNN(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_rnnlayers, n_outputs):
        super(RNN, self).__init__()
        self.D = n_inputs
        self.M = n_hidden
        self.K = n_outputs
        self.L = n_rnnlayers

        self.rnn = nn.LSTM(
            input_size=self.D,
            hidden_size=self.M,
            num_layers=self.L,
            batch_first=True,
            dropout=0.5)
        self.fc = nn.Linear(self.M, self.K)
    
    def forward(self, X):
        h0 = torch.zeros(self.L, X.size(0), self.M)
        c0 = torch.zeros(self.L, X.size(0), self.M)
        out, _ = self.rnn(X, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def full_gd(model, criterion, optimizer,  X_train, y_train, X_test,  y_test, epochs=500):

    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)

    for it in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        train_losses [it] = loss.item()
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        test_losses[it] = test_loss.item()


        if (it + 1) % 5 == 0:
            print(f'Epoch {it + 1} / {epochs}, Train Loss: {loss.item(): .4f}, Test Loss : {test_loss.item(): .4f}')
    print(statistics.mean(test_losses))
    print(statistics.mean(train_losses))

    return train_losses, test_losses

def ThirdtApproachLSTM(file_path, time_window, num_hidden_states, num_hidden_layers, learning_rate):
    df = pd.read_csv(file_path)
    df = df.drop([0, 1,2, 3])
    input_data = df[['Price', 'SMA', 'EMA', 'RSI']].values
    targets = df['Price_Return'].values
    T = time_window
    D = input_data.shape[1]
    N = len(input_data) - T
    N_train = 2 * len(input_data) // 3
    scaler = StandardScaler()
    scaler.fit(input_data[: N_train + T -1])
    input_data = scaler.transform(input_data)
    X_train = np.zeros((N_train, T, D))
    Y_train = np.zeros((N_train, 1))
    for i in range(N_train):
        X_train[i, :,:] = input_data[i:i+T]
        Y_train[i] = (targets[i+T] > 0)
    X_test = np.zeros((N - N_train, T, D))
    Y_test = np.zeros((N - N_train, 1))
    for j in range(N - N_train):
        t = j + N_train
        X_test[j, :,:] = input_data[t:t+T]
        Y_test[j] = (targets[t+T] > 0)
    model = RNN(4, num_hidden_states, num_hidden_layers, 1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_losses, test_losses = full_gd(model, criterion, optimizer, X_train, Y_train, X_test, Y_test)
    plt.plot(train_losses, label='train loss')
    plt.plot(test_losses, label='test loss')
    plt.rcParams['figure.figsize'] = 25, 6
    plt.legend()
    plt.show()
    with torch.no_grad():
        p_train = model(X_train)
        p_train = (p_train.cpu().numpy() > 0)
        train_acc = np.mean(Y_train.cpu().numpy()  == p_train)

        p_test = model(X_test)
        p_test = (p_test.cpu().numpy() > 0)
        test_acc = np.mean(Y_test.cpu().numpy()  == p_test)
    print(f'Train_acc: {train_acc: .4f}, Test_acc: {test_acc: .4f}')


def ThirdApproachTree(stockname):
    df = pd.read_csv(stockname)
    del df['Unnamed: 0']	
    df = df.drop([0, 1,2, 3])
    plt.plot(df['Price'])
    df.columns
    features = ['Price', 'SMA', 'EMA', 'RSI']
    x = df[features]
    y = np.where(df.Price_Return > 0, 1, 0)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=77)
    treeClassifier = DecisionTreeClassifier(max_depth=3, min_samples_leaf=6)
    treeClassifier.fit(X_train, y_train)
    y_pred = treeClassifier.predict(X_test)
    report = classification_report(y_test, y_pred)
    data = tree.export_graphviz(treeClassifier, filled=True, feature_names=features, class_names=np.array(['Down','UP']))
    graphviz.Source(data)

def FirstApproachLSTM(file_path, stock_name, time_window, num_hidden_states, num_hidden_layers, larning_rate):
    df = pd.read_csv(file_path)
    price = df[stock_name].values.reshape(-1,1)
    scaler = StandardScaler()
    scaler.fit(price[: (1 * len(price)) // 3])
    price = scaler.transform(price).flatten()
    T = time_window
    D = 1
    X = []
    Y = []
    for t in range(len(price) - T):
        x = price[t:t+T]
        X.append(x)
        y = price[t+T]
        Y.append(y)
    X = np.array(X).reshape(-1,T,1)
    Y = np.array(Y).reshape(-1,1)
    N = len(X)
    model = RNN(1, num_hidden_states, num_hidden_layers, 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=larning_rate)
    X_train = torch.from_numpy(X[ :(-1 * N)//3].astype(np.float32))
    Y_train = torch.from_numpy(Y[:(-1 * N)//3].astype(np.float32))
    X_test = torch.from_numpy(X[(-1 * N)//3:].astype(np.float32))
    Y_test = torch.from_numpy(Y[(-1 * N)//3:].astype(np.float32))   
    train_losses, test_losses = full_gd(model, criterion, optimizer, X_train, Y_train, X_test, Y_test)
    plt.plot(train_losses, label='train loss')
    plt.plot(test_losses, label='test loss')
    plt.legend()
    plt.show()
    validation_target = Y
    validation_prediction = []
    i = 0
    in_ = torch.from_numpy(X.astype(np.float32))
    while len(validation_prediction) < len(validation_target):
        input_ = in_[i].reshape(1,T, 1)
        p = model(input_)[0,0].item()
        i += 1
        validation_prediction.append(p)
    plt.plot(validation_target, label='forcast target')
    plt.plot(validation_prediction, label='forcast prediction')
    plt.legend()
    plt.show()

def SecondApproachLSTM(file_path, stock_name, time_window, num_hidden_states, num_hidden_layers, learning_rate):
    df = pd.read_csv(file_path)
    price = df[stock_name].shift(1)
    return_ = (df[stock_name] - price) / price
    series = return_.values[1:].reshape(-1, 1)
    scaler = StandardScaler()
    scaler.fit(series[: (2 * len(series)) // 3])
    series = scaler.transform(series).flatten()
    T = time_window
    D = 1
    X = []
    Y = []
    for t in range(len(series) - T):
        x = series[t:t+T]
        X.append(x)
        y = series[t+T]
        Y.append(y)
    X = np.array(X).reshape(-1,T,1)
    Y = np.array(Y).reshape(-1,1)
    N = len(X)
    model = RNN(1, num_hidden_states, num_hidden_layers, 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    X_train = torch.from_numpy(X[ :(-1 * N)//3].astype(np.float32))
    Y_train = torch.from_numpy(Y[:(-1 * N)//3].astype(np.float32))
    X_test = torch.from_numpy(X[(-1 * N)//3:].astype(np.float32))
    Y_test = torch.from_numpy(Y[(-1 * N)//3:].astype(np.float32))
    train_losses, test_losses = full_gd(model, criterion, optimizer, X_train, Y_train, X_test, Y_test)
    plt.plot(train_losses, label='train loss')
    plt.plot(test_losses, label='test loss')
    plt.rcParams['figure.figsize'] = 25, 6
    plt.legend()
    plt.show()
    validation_target = Y
    validation_prediction = []
    i = 0
    in_ = torch.from_numpy(X.astype(np.float32))
    while len(validation_prediction) < len(validation_target):
        input_ = in_[i].reshape(1,T, 1)
        p = model(input_)[0,0].item()
        i += 1
        validation_prediction.append(p)
    plt.plot(validation_target, label='forcast target')
    plt.plot(validation_prediction, label='forcast prediction')
    plt.legend()
    plt.show()





def main():
    #FirstApproachLSTM()
    #SecondApproachLSTM()
    #ThirdApproachTree()
    ThirdtApproachLSTM()




if __name__ == "__main__":
    main()

