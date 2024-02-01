import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import glob


def load_target_data(filepath):
    """
    Give individual target filepath
    """
    cols = ["Bbox Center X", "Bbox Center Y",
            "BBox Width", "Bbox Height",
            "Yaw", "Pitch",
            "Gaze Target Index - X",
            "Gaze Target Index - Y",
            "Gaze Target - U",
            "Gaze Target - V"]
    data_list = []
    x = np.load(filepath)
    data_list.extend(x)
    data_df = pd.DataFrame(data_list, columns=cols)
    # Convert index values into ints
    for i in ["Gaze Target Index - X", "Gaze Target Index - Y"]:
        data_df[i] = data_df[i].astype(int)
    # Convert pitch and yaw into degrees, for human readability
    for i in ["Yaw", "Pitch"]:
        data_df[i] = data_df[i] * 180. / np.pi

    # Drop U/V values
    data_cols = ["Bbox Center X", "Bbox Center Y",
                 "BBox Width", "Bbox Height",
                 "Yaw", "Pitch",
                 "Gaze Target Index - X", "Gaze Target Index - Y"]

    return data_df[data_cols]


def distance_from_median(input_df, col_name, median_threshold):
    # Return index of items outside the threshold bound to drop
    median = input_df.median()[col_name]
    return input_df[abs(input_df[col_name] - median) > median_threshold].index


def less_than(input_df, col_name, less_than_threshold):
    return input_df[input_df[col_name] < less_than_threshold].index


def greater_than(input_df, col_name, greater_than_threshold):
    return input_df[input_df[col_name] > greater_than_threshold].index


def clean_angular_data(session_name,
                       criteria_col="Yaw",
                       criteria_threshold=15 * np.pi / 180.,
                       criteria=distance_from_median):
    # Criteria returns index of elements to drop
    cols = ["Bbox Center X", "Bbox Center Y",
            "BBox Width", "Bbox Height",
            "Yaw", "Pitch",
            "Gaze Target Index - X",
            "Gaze Target Index - Y",
            "Gaze Target - U",
            "Gaze Target - V"]
    for col in range(4):
        for row in range(4):
            target_filepath = f"./calibration_data/{session_name}/cleaned_target_{col}_{row}.npy"
            data_list = []
            data_list.extend(np.load(target_filepath))
            df = pd.DataFrame(data_list, columns=cols)
            index = criteria(df, criteria_col, criteria_threshold)
            cleaned_df = df.drop(index)
            cleaned_filepath = f"./calibration_data/{session_name}/15_{criteria_col}_cleaned_target_{col}_{row}.npy"
            np.save(cleaned_filepath, cleaned_df.to_numpy())


def clean_target_data(session_name, criteria_col, criteria_threshold, less_than=True):
    cols = ["Bbox Center X", "Bbox Center Y",
            "BBox Width", "Bbox Height",
            "Yaw", "Pitch",
            "Gaze Target Index - X",
            "Gaze Target Index - Y",
            "Gaze Target - U",
            "Gaze Target - V"]
    for col in range(4):
        for row in range(4):
            target_filepath = f"./calibration_data/{session_name}/target_{col}_{row}.npy"
            data_list = []
            data_list.extend(np.load(target_filepath))
            df = pd.DataFrame(data_list, columns=cols)
            if less_than:
                index = df[df[criteria_col] < criteria_threshold].index
            else:
                index = df[df[criteria_col] > criteria_threshold].index
            cleaned_df = df.drop(index)
            cleaned_filepath = f"./calibration_data/{session_name}/cleaned_target_{col}_{row}.npy"
            np.save(cleaned_filepath, cleaned_df.to_numpy())


def plot_session_data(session_name, x_col="Bbox Center X",
                      y_col="Bbox Center Y",
                      x_lim=(0, 1920),
                      y_lim=(0, 1200),
                      clean_prefix=""):
    fig, axes = plt.subplots(4, 4, sharex=True, sharey=True)
    # Setting the values for all axes.
    plt.setp(axes, xlim=x_lim, ylim=y_lim)
    fig.suptitle(session_name)

    for col in range(4):
        for row in range(4):
            target_filepath = f"./calibration_data/{session_name}/{clean_prefix}target_{col}_{row}.npy"
            target_data = load_target_data(target_filepath)
            # data is stored with col_row.npy form, so row and col are flipped to make subplots line up with data points
            sns.scatterplot(x=target_data[x_col], y=target_data[y_col], ax=axes[row, col]).set_title(
                f"row={row}, col={col}")


def plot_aggregate_session_data(session_name,
                                x_col="Bbox Center X",
                                y_col="Bbox Center Y",
                                x_lim=(0, 1920),
                                y_lim=(0, 1200),
                                clean_prefix=""):
    for col in range(4):
        for row in range(4):
            ind_file = f"./calibration_data/{session_name}/{clean_prefix}target_{col}_{row}.npy"
            target_data = load_target_data(ind_file)
            if col == 0 and row == 0:
                agg_df = target_data
            else:
                agg_df = pd.concat([agg_df, target_data], ignore_index=True)
    plt.figure()
    sns.scatterplot(data=agg_df, x=x_col, y=y_col).set_title(session_name)


# plt.ion()
calibration_data_dir = os.path.join(os.getcwd(), "calibration_data/")
session_names = glob.glob("2024*", root_dir=calibration_data_dir)

criteria_list = {'20240131-193321': ("Bbox Center Y", False, 800),
                 '20240131-195237': ("Bbox Center X", True, 800),
                 '20240131-195952': ("Bbox Center X", True, 0),
                 '20240131-200507': ("Bbox Center X", True, 0),
                 '20240131-201217': ("Bbox Center X", True, 0),
                 '20240131-201733': ("Bbox Center X", True, 0),
                 '20240131-211542': ("Bbox Center X", True, 400),
                 '20240131-212331': ("Bbox Center X", True, 100),
                 '20240131-212915': ("Bbox Center X", True, 0),
                 '20240131-214227': ("Bbox Center X", True, 400),
                 '20240131-220105': ("Bbox Center X", True, 400),
                 '20240131-221849': ("Bbox Center X", True, 0),
                 '20240131-222644': ("Bbox Center X", True, 400),
                 '20240131-223332': ("Bbox Center X", True, 0),
                 }
clean_prefix = ""
clean_data = False
plot_session = True
plot_aggregate = False
plot_options = {"Bbox X-Y": [("Bbox Center X", "Bbox Center Y"), [(0, 1920), (0, 1200)]],
                "Yaw-Pitch": [("Yaw", "Pitch"), [(-180, 180), (-180, 180)]]}
plot_data = plot_options["Yaw-Pitch"]

for s_name in session_names:
    if clean_data:
        # crit_name, less, threshold = criteria_list[s_name]
        # clean_target_data(s_name, crit_name, less_than=less, criteria_threshold=threshold)
        clean_angular_data(s_name)
    else:
        if plot_session:
            plot_session_data(s_name, x_col=plot_data[0][0], y_col=plot_data[0][1],
                              x_lim=plot_data[1][0],
                              y_lim=plot_data[1][1],
                              clean_prefix=clean_prefix)
        if plot_aggregate:
            plot_aggregate_session_data(s_name, x_col=plot_data[0][0], y_col=plot_data[0][1], clean_prefix=clean_prefix)

plt.show()

# Next up, take cleaned data and look at Yaw/Pitch distributions for points in a session
