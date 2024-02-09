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
                       criteria_threshold=35 * np.pi / 180.,
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
            cleaned_filepath = f"./calibration_data/{session_name}/35_{criteria_col}_cleaned_target_{col}_{row}.npy"
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
session_names = glob.glob("20240201-21*", root_dir=calibration_data_dir)

# criteria_list = {'20240131-193321': ("Bbox Center Y", False, 800),
#                  '20240131-195237': ("Bbox Center X", True, 800),
#                  '20240131-195952': ("Bbox Center X", True, 0),
#                  '20240131-200507': ("Bbox Center X", True, 0),
#                  '20240131-201217': ("Bbox Center X", True, 0),
#                  '20240131-201733': ("Bbox Center X", True, 0),
#                  '20240131-211542': ("Bbox Center X", True, 400),
#                  '20240131-212331': ("Bbox Center X", True, 100),
#                  '20240131-212915': ("Bbox Center X", True, 0),
#                  '20240131-214227': ("Bbox Center X", True, 400),
#                  '20240131-220105': ("Bbox Center X", True, 400),
#                  '20240131-221849': ("Bbox Center X", True, 0),
#                  '20240131-222644': ("Bbox Center X", True, 400),
#                  '20240131-223332': ("Bbox Center X", True, 0),
#                  }

# criteria_list = {'20240201-194839': ("Bbox Center X", True, 400),
#                  '20240201-193453': ("Bbox Center Y", False, 600),
#                  '20240201-195417': ("Bbox Center X", True, 0),
#                  '20240201-194216': ("Bbox Center X", True, 0),
#                  '20240201-192719': ("Bbox Center X", True, 0),
#                  }

criteria_list = {'20240201-213457': ("Bbox Center X", True, 500),
                 '20240201-211913': ("Bbox Center X", True, 600),
                 '20240201-214159': ("Bbox Center X", False, 940),
                 '20240201-212858': ("Bbox Center X", True, 0)
                 }

## OK, leaving a note here on the hacked together process for cleaning data before I forget or move on
# 1. Get list of  folders to examine/clean (session_names). Can be directly set or based on glob and search pattern
# 2. Set clean_data and plot_session to False, plot_aggregate to True. Want to determine cutoff threshold for detections
#    of multiple faces
# 3. Build criteria_list with cutoff params and values. ("Bbox Center X", True, 500) rejects all points less than 500
# 4. Uncomment (ugly, I know) the lines for crit_name, clean_target_data under if clean_data.
#    Comment out clean_angular_data. If we take the project forward, I'll re-write this.
# 5. With cleaned_ files generated, switch back from step 4. Run clean_angular_data.
# 6. Should have cleaned data with 35_Yaw_cleaned_ prefix now. Can be used for training. 35 refers to the angular band,
#    in degrees, that will be filtered out
# Note: This assumes each session has a person standing in one place, staring at a specific dot.

clean_prefix = "35_Yaw_cleaned_"
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
