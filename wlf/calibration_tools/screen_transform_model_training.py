import argparse
import copy
from pathlib import Path
import os
import glob
import joblib
import matplotlib
matplotlib.use('TkAgg')
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
from wlf.calibration_tools.regression_nn import RegressionNeuralNetwork, ClassificationNeuralNetwork


#TODO: Try out classifier model with 8 bins
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Model training to map from detected gaze vector to screen pierce point')
    parser.add_argument(
        '--data_timestr', dest='data_timestr', help='Timestring for training data',
        default="2024", type=str)
    parser.add_argument(
        '--model_dir', dest='model_dir', help='Relative path for model directory',
        default="calibration_models", type=str)
    parser.add_argument(
        '--scale_data', dest='scale_data', default=True, action='store_true')
    parser.add_argument(
        '--classification', dest='classification', default=False, action='store_true')

    return parser.parse_args()


def load_data(data_timestring, classification=True, cleaned_prefix="", directory_list=None):
    """
    Give path relative to calibration_data folder
    """
    calibration_data_dir = os.path.join(os.getcwd(), "calibration_data")
    cols = ["Bbox Center X", "Bbox Center Y",
            "BBox Width", "Bbox Height",
            "Yaw", "Pitch",
            "Gaze Target Index - X",
            "Gaze Target Index - Y",
            "Gaze Target - U",
            "Gaze Target - V"]
    data_list = []
    if directory_list is None:
        directory_list = glob.glob(calibration_data_dir + f"/{data_timestring}*")
    else:
        directory_list = [calibration_data_dir + f"/{i}" for i in directory_list]
    # Get all folders matching data_timestring. Should probably allow specification of search string
    for data_dir in directory_list:
        for col in range(4):
            for row in range(4):
                x = np.load(os.path.join(data_dir, f"{cleaned_prefix}target_{col}_{row}.npy"))
                data_list.extend(x.tolist())
    data_df = pd.DataFrame(data_list, columns=cols)
    data_cols = ["Bbox Center X", "Bbox Center Y",
                 "BBox Width", "Bbox Height",
                 "Yaw", "Pitch"]
    for i in ["Gaze Target Index - X", "Gaze Target Index - Y"]:
        data_df[i] = data_df[i].astype(int)
    target_cols = ["Gaze Target Index - X"] if classification else ["Gaze Target - U"]
    return data_df[data_cols].to_numpy(), data_df[target_cols].to_numpy()


def get_model(classification=True, learning_rate=1e-3, model_weight_decay=1e-4):
    if classification:
        nn_model = ClassificationNeuralNetwork().to(device)
        print(nn_model)
        loss_function = nn.CrossEntropyLoss()
        nn_optimizer = optim.Adam(nn_model.parameters(), lr=learning_rate, weight_decay=model_weight_decay)
    else:
        nn_model = RegressionNeuralNetwork().to(device)
        print(nn_model)
        # loss function and optimizer
        loss_function = nn.MSELoss()  # mean square error
        nn_optimizer = optim.Adam(nn_model.parameters(), lr=learning_rate, weight_decay=model_weight_decay)
    return nn_model, loss_function, nn_optimizer


if __name__ == '__main__':
    args = parse_args()
    train_timestr = time.strftime("%Y%m%d-%H%M%S")
    data_timestr = args.data_timestr
    scale_data = args.scale_data
    model_dir = args.model_dir
    classification = args.classification

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # Read data
    # good_dir = ['20240131-212915', '20240131-195237', '20240131-212331',
    #             '20240131-220105', '20240131-221849', '20240131-222644']
    # mid_dir = ['20240131-211542', '20240131-201733', '20240131-201217']
    # poor_dir = []
    full_set = ['20240201-192719', '20240201-193453', '20240201-194216', '20240201-194839', '20240201-195417',
                '20240131-193321', '20240131-195237', '20240131-195952', '20240131-200507', '20240131-201217',
                '20240131-201733', '20240131-211542', '20240131-212331', '20240131-212915', '20240131-214227',
                '20240131-220105', '20240131-221849', '20240131-222644', '20240131-223332', '20240201-213457',
                '20240201-211913', '20240201-214159', '20240201-212858']

    data, target = load_data(data_timestr, classification, cleaned_prefix="35_Yaw_cleaned_", directory_list=full_set)
    X, y = data, target

    if scale_data:
        sc = StandardScaler()
        X = sc.fit_transform(X)
        Path(f"./{model_dir}").mkdir(exist_ok=True)
        joblib.dump(sc, f"{model_dir}\\{train_timestr}_{data_timestr}_scalar.bin", compress=True)

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
    if classification:
        y_train = y_train.flatten().to(int)
        y_test = y_test.flatten().to(int)

    model, loss_fn, optimizer = get_model(classification)

    n_epochs = 10000  # number of epochs to run
    batch_size = 4096  # size of each batch
    batch_start = torch.arange(0, len(X_train), batch_size)

    # Hold the best model
    best_error = np.inf  # init to infinity
    best_weights = None
    best_acc = 0.
    train_loss_history = []
    test_loss_history = []
    train_acc_history = []
    test_acc_history = []

    for epoch in range(n_epochs):
        epoch_loss = []
        epoch_acc = []
        model.train()
        # print(f"Epoch: {epoch}")
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=False) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # take a batch
                X_batch = X_train[start:start + batch_size]
                y_batch = y_train[start:start + batch_size]

                # forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress and store metrics
                if classification:
                    acc = float((torch.argmax(y_pred, 1) == y_batch).sum())/len(y_batch)
                else:
                    acc = 0.
                epoch_acc.append(acc)
                epoch_loss.append(float(loss))
                bar.set_postfix(error=float(loss),
                                acc=acc)
        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(X_test)
        test_loss = loss_fn(y_pred, y_test)

        test_loss = float(test_loss)
        if classification:
            test_acc = float((torch.argmax(y_pred, 1) == y_test).sum()) / len(y_test)
            # If interested in filtering based on confidence:
            # threshold = 0.50
            # y_pred_probs = torch.softmax(y_pred, dim=1)
            # conf_index = (y_pred_probs.max(dim=1).values > threshold)
            # percent_above_threshold = (y_pred_probs.max(dim=1).values > threshold).sum() / len(y_pred_probs)
            # float((torch.argmax(y_pred[conf_index], 1) == y_test[conf_index]).sum()) / len(y_test[conf_index])
            # for i in range(10):
            #     conf_threshold = 0.1 * i
            #     conf_index = (y_pred_probs.max(dim=1).values > conf_threshold)
            #     count_above_threshold = (y_pred_probs.max(dim=1).values > conf_threshold).sum()
            #     percent_above_threshold = (y_pred_probs.max(dim=1).values > conf_threshold).sum() / len(y_pred_probs)
            #     test_acc = float((torch.argmax(y_pred[conf_index], 1) == y_test[conf_index]).sum()) / len(
            #         y_test[conf_index])
            #     print(
            #         f"conf_threshold: {conf_threshold:.2f} Perc above thresh: {percent_above_threshold:.3f} test Acc: {test_acc:.3f}")

        else:
            test_acc = 0.

        test_loss_history.append(test_loss)
        test_acc_history.append(test_acc)
        train_loss_history.append(np.mean(epoch_loss))
        train_acc_history.append(np.mean(epoch_acc))
        if test_loss < best_error:
            best_error = test_loss
            best_weights = copy.deepcopy(model.state_dict())
            best_acc = test_acc

    # restore model and return best accuracy
    model.load_state_dict(best_weights)

    torch.save(model.state_dict(), os.path.join(os.getcwd(), f"{model_dir}\\{train_timestr}_{data_timestr}_model.ckpt"))
    print("Error: %.2f" % best_error)
    print("RM Error: %.2f" % np.sqrt(best_error))
    print("Best Model Acc: %.2f" % best_acc)

    error_type = "Cross Entropy" if classification else "MSE"
    plt.plot(train_loss_history, label="train")
    plt.plot(test_loss_history, label="test")
    plt.xlabel("Epochs")
    plt.ylabel(error_type)
    plt.legend()
    plt.show()

    plt.plot(train_acc_history, label="train")
    plt.plot(test_acc_history, label="test")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
