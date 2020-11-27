# ml_network_0.1.py
# ML solution for dss version 0.1
# https://pytorch.org/docs/stable/index.html
# Nov 2020 

from nn.subs import SubsDS
from nn.nets import SubsNN, SubsNN2, FFNN
import numpy as np
import torch as T
import time
import sys

# Declare a device
device = T.device("cpu")


def accuracy(model, ds, pct):
    """accuracy
    copied from
    https://jamesmccaffrey.wordpress.com/2020/10/13/regression-using-pytorch/

    Args:
            model (T.nn.Module): [description]
            ds ([type]): [description]
            pct ([type]): [description]

    Returns:
            [type]: [description]
    """
    # assumes model.eval()
    # percent correct within pct of true income
    n_correct = 0
    n_wrong = 0

    for i in range(len(ds)):
        X = ds[i]['subs']             # 2-d
        Y = ds[i]['duration']         # 2-d
        with T.no_grad():
            oupt = model(X)         # computed income

        if np.abs(oupt.item() - Y.item()) < np.abs(pct * Y.item()):
            n_correct += 1
        else:
            n_wrong += 1

    acc = (n_correct * 1.0) / (n_correct + n_wrong)
    return acc


def load_dss():
    print("Load start ---- ")

    T.manual_seed(1)
    np.random.seed(1)
    
    print("Loading dss")
    train_file = "subs_dss_0.1_sorted_norm.csv"
    train_ds = SubsDS(train_file)

    test_file = "subs_dss_0.1_sorted_norm_test.csv"
    test_ds = SubsDS(test_file)

    batch_size = 50
    train_loader = T.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True)
    
    print(train_loader.__dict__)
    return train_loader, train_ds, test_ds
    

def train(net, train_loader):
    max_epochs = 2000
    ep_log_interval = 100
    lrn_rate = 0.01

    loss_func = T.nn.MSELoss()
    optimizer = T.optim.SGD(net.parameters(), lr=lrn_rate)

    print(f"batch_size: {train_loader.batch_size}")
    print(f"loss: {loss_func}")
    print(f"optimizer: SGD")
    print(f"max_epochs: {max_epochs}")
    print(f"lrn_rate: {lrn_rate}")

    print("Start training.")
    net = net.train()
    for epoch in range(0, max_epochs):
        epoch_loss = 0                  # for one full epoch
        for (batch_idx, batch) in enumerate(train_loader):
            X = batch['subs']           # [10,8]
            Y = batch['duration']       # [10,1]   
            optimizer.zero_grad()
            oupt = net(X)                  # predicted income
            loss_val = loss_func(oupt, Y)  # a tensor
            epoch_loss += loss_val.item()  # accumulate
            loss_val.backward()
            optimizer.step()

        if epoch % ep_log_interval == 0:
            print(f"epoch: {epoch}    loss: {epoch_loss:.4f}")

    print("Done.")
    return net


def save(net, name="ml_subs_0.1_model.pth"):
    """Save model to file

    Args:
        net (T.nn.Module):
    """
    print("Saving model state...", end=" ")
    T.save(net.state_dict(), name)
    print("Ok")


def measure_accuracy(net, train_ds, test_ds):
    print("Computing model accuracy")
    net = net.eval()
    acc_train = accuracy(net, train_ds, 0.10)  # item-by-item
    print(f"Accuracy on training data: {acc_train:0.4f}")

    acc_test = accuracy(net, test_ds, 0.10)  # item-by-item
    print(f"Accuracy on test data: {acc_test:0.4f}")

    # 5. make a prediction
    print("Predicting duration for D(20001:20002, :): ")
    unk = np.array([[3.00, 1.00, 1.00, 6.00, 7.00, 2.00, 2.00]], dtype=np.float32)
    unk = T.tensor(unk, dtype=T.float32).to(device) 

    with T.no_grad():
        pred_inc = net(unk)
    pred_inc = pred_inc.item()  # scalar
    print(f"${(pred_inc):.2f}")


if __name__ == "__main__":
    s = time.time()
    tl, train_ds, test_ds = load_dss()

    if '-l' in sys.argv:
        fname = sys.argv[sys.argv.index('-l') + 1]
        net = SubsNN2().to(device)
        net.load_state_dict(T.load(fname))
        net.eval()
    else:
        if '-p' in sys.argv:
            lp = sys.argv[sys.argv.index('-p') + 1]
            layer_conf = lp.split(",") if "," in lp else lp.split(" ")
            layer_conf = map(lambda k: int(k), layer_conf)
        else:
            layer_conf = (7, 1, 7)

        net = FFNN(*layer_conf).to(device)
        net = train(net, tl)
        print(f"train: {time.time()-s:.2f}s.")

    s1 = time.time()
    if '-s' in sys.argv:
        save(net, f"ml_subs_0.1.1_{net.print()}_{int(time.time())}.pth")
        print(f"saving: {time.time()-s1:.2f}s.")
    s2 = time.time()
    measure_accuracy(net, train_ds, test_ds)
    print(f"accuracy: {time.time()-s2:.2f}s.")
