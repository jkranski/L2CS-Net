import argparse
import copy
from pathlib import Path
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from wlf.calibration_tools.regression_nn import RegressionNeuralNetwork


#TODO: Try out classifier model with 8 bins
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Model training to map from detected gaze vector to screen pierce point')
    parser.add_argument(
        '--data_timestr', dest='data_timestr', help='Timestring for training data',
        default="20231215-154754", type=str)
    parser.add_argument(
        '--model_dir', dest='model_dir', help='Relative path for model directory',
        default="calibration_models", type=str)
    parser.add_argument(
        '--scale_data', dest='scale_data', default=True, action='store_true')

    return parser.parse_args()


def load_data(data_timestring):
    """
    Give path relative to calibration_data folder
    """
    full_path = os.path.join(os.getcwd(), "calibration_data")
    full_path = os.path.join(full_path, data_timestring)
    cols = ["Bbox Center X", "Bbox Center Y",
            "BBox Width", "Bbox Height",
            "Yaw", "Pitch",
            "Gaze Target Index - X",
            "Gaze Target Index - Y",
            "Gaze Target - U",
            "Gaze Target - V"]
    data_list = []
    for i in range(4):
        for j in range(4):
            x = np.load(os.path.join(full_path, f"target_{i}_{j}.npy"))
            data_list.extend(x.tolist())
    data_df = pd.DataFrame(data_list, columns=cols)
    data_cols = ["Bbox Center X", "Bbox Center Y",
                 "BBox Width", "Bbox Height",
                 "Yaw", "Pitch"]
    target_cols = ["Gaze Target - U"]
    return data_df[data_cols].to_numpy(), data_df[target_cols].to_numpy()


if __name__ == '__main__':
    args = parse_args()
    train_timestr = time.strftime("%Y%m%d-%H%M%S")
    data_timestr = args.data_timestr
    scale_data = args.scale_data
    model_dir = args.model_dir

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # Read data
    data, target = load_data(data_timestr)
    X, y = data, target

    if scale_data:
        sc = StandardScaler()
        X = sc.fit_transform(X)
        Path(f"./{model_dir}").mkdir(exist_ok=True)
        joblib.dump(sc, f"{model_dir}\\{train_timestr}_{data_timestr}_scalar.bin", compress=True)
        #TODO: Don't hardcode screen width
        #y = y/1920.

    # train-test split for model evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True)

    # Convert to 2D PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
    # Convert to 2D PyTorch tensors
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    model = RegressionNeuralNetwork().to(device)
    print(model)

    # loss function and optimizer
    loss_fn = nn.MSELoss()  # mean square error
    optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-2)

    n_epochs = 10000  # number of epochs to run
    batch_size = 4096  # size of each batch
    batch_start = torch.arange(0, len(X_train), batch_size)

    # Hold the best model
    best_mse = np.inf  # init to infinity
    best_weights = None
    history = []

    for epoch in range(n_epochs):
        model.train()
        # print(f"Epoch: {epoch}")
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=False) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # take a batch
                X_batch = X_train[start:start + batch_size]
                y_batch = y_train[start:start + batch_size]
                # Move to device
                #X_batch = X_batch.to(device)
                #y_batch = y_batch.to(device)
                # forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)

                #u_pred, v_pred = model(X_batch)
                #loss_u = loss_fn(u_pred, y_batch[:, 0])
                #loss_v = loss_fn(v_pred, y_batch[:, 1])
                #loss = loss_u + loss_v
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress
                bar.set_postfix(mse=float(loss))
        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(X_test)
        mse = loss_fn(y_pred, y_test)
        #u_pred, v_pred = model(X_test)
        #loss_u = loss_fn(u_pred, y_test[:, 0])
        #loss_v = loss_fn(v_pred, y_test[:, 1])
        #mse = loss_u + loss_v
        mse = float(mse)
        history.append(mse)
        #print(f"MSE: {mse}")
        if mse < best_mse:
            best_mse = mse
            best_weights = copy.deepcopy(model.state_dict())

    # restore model and return best accuracy
    model.load_state_dict(best_weights)

    torch.save(model.state_dict(), os.path.join(os.getcwd(), f"{model_dir}\\{train_timestr}_{data_timestr}_model.ckpt"))
    print("MSE: %.2f" % best_mse)
    print("RMSE: %.2f" % np.sqrt(best_mse))
    plt.plot(history)
    plt.show()
