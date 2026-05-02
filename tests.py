import scipy.stats as ss
import os
import sys
import numpy as np
import pandas as pd

import scipy.stats as ss
import os
import numpy as np
import pandas as pd

import scipy.stats as ss
import os
import numpy as np
import pandas as pd

# store all results
results_dict = {}

methods = ["PS", "MI", "LogISI", "CMA"]

# synthetic sens/spec
analysis_path = os.path.join('thesis', 'analysis_df.csv')
analysis_df = pd.read_csv(analysis_path)

# iterate through metrics and methods
for metric in ["sens", "1spec"]:
    for method in methods:

        unified_col = analysis_df[f"Unified_{metric}"].values
        method_col = analysis_df[f"{method}_{metric}"].values

        # do two-sided t-test
        t_stat, p_val = ss.ttest_rel(unified_col, method_col, nan_policy='omit')

        # save to dictionary
        key = f"{metric}_{method}"
        results_dict[f"{key}_Unified_mean"] = np.nanmean(unified_col)
        results_dict[f"{key}_Unified_sd"] = np.nanstd(unified_col, ddof=1)
        results_dict[f"{key}_Method_mean"] = np.nanmean(method_col)
        results_dict[f"{key}_Method_sd"] = np.nanstd(method_col, ddof=1)
        results_dict[f"{key}_p"] = p_val
        results_dict[f"{key}_t"] = t_stat


# synthetic burst rate
frame_path = os.path.join('thesis', 'data_frame.csv')
df = pd.read_csv(frame_path)

for method in methods:
    unified_col = df[f"unified_burst_rate"].values
    method_col = df[f"{method.lower()}_burst_rate"].values

    t_stat, p_val = ss.ttest_rel(unified_col, method_col, nan_policy='omit')

    key = f"burst_rate_{method}"
    results_dict[f"{key}_Unified_mean"] = np.nanmean(unified_col)
    results_dict[f"{key}_Unified_sd"] = np.nanstd(unified_col, ddof=1)
    results_dict[f"{key}_Method_mean"] = np.nanmean(method_col)
    results_dict[f"{key}_Method_sd"] = np.nanstd(method_col, ddof=1)
    results_dict[f"{key}_p"] = p_val
    results_dict[f"{key}_t"] = t_stat

# pd burst rate
pd_frame_path = os.path.join('thesis', 'pd_data_frame.csv')
pd_df = pd.read_csv(pd_frame_path)

for method in methods:

    unified_col = pd_df[f"unified_burst_rate"].values
    method_col = pd_df[f"{method.lower()}_burst_rate"].values

    t_stat, p_val = ss.ttest_rel(unified_col, method_col, nan_policy='omit')

    key = f"pd_burst_rate_{method}"
    results_dict[f"{key}_Unified_mean"] = np.nanmean(unified_col)
    results_dict[f"{key}_Unified_sd"] = np.nanstd(unified_col, ddof=1)
    results_dict[f"{key}_Method_mean"] = np.nanmean(method_col)
    results_dict[f"{key}_Method_sd"] = np.nanstd(method_col, ddof=1)
    results_dict[f"{key}_p"] = p_val
    results_dict[f"{key}_t"] = t_stat

# synthetic % spikes in burst
frame_path = os.path.join('thesis', 'data_frame.csv')
df = pd.read_csv(frame_path)

for method in methods:
    unified_col = df[f"unified_%_spikes_in_burst"].values
    method_col = df[f"{method.lower()}_%_spikes_in_burst"].values

    t_stat, p_val = ss.ttest_rel(unified_col, method_col, nan_policy='omit')

    key = f"spikes_in_burst_{method}"
    results_dict[f"{key}_Unified_mean"] = np.nanmean(unified_col)
    results_dict[f"{key}_Unified_sd"] = np.nanstd(unified_col, ddof=1)
    results_dict[f"{key}_Method_mean"] = np.nanmean(method_col)
    results_dict[f"{key}_Method_sd"] = np.nanstd(method_col, ddof=1)
    results_dict[f"{key}_p"] = p_val
    results_dict[f"{key}_t"] = t_stat

# pd burst rate
pd_frame_path = os.path.join('thesis', 'pd_data_frame.csv')
pd_df = pd.read_csv(pd_frame_path)

for method in methods:

    unified_col = pd_df[f"unified_%_spikes_in_burst"].values
    method_col = pd_df[f"{method.lower()}_%_spikes_in_burst"].values

    t_stat, p_val = ss.ttest_rel(unified_col, method_col, nan_policy='omit')

    key = f"pd_spikes_in_burst_{method}"
    results_dict[f"{key}_Unified_mean"] = np.nanmean(unified_col)
    results_dict[f"{key}_Unified_sd"] = np.nanstd(unified_col, ddof=1)
    results_dict[f"{key}_Method_mean"] = np.nanmean(method_col)
    results_dict[f"{key}_Method_sd"] = np.nanstd(method_col, ddof=1)
    results_dict[f"{key}_p"] = p_val
    results_dict[f"{key}_t"] = t_stat

# save to file
path = os.path.join('thesis', 'ttests.txt')
array = np.array(list(results_dict.items()), dtype=object)
np.savetxt(path, array, fmt="%s", delimiter=":")

# calculate % difference between healthy and dd groups for metrics
def pd_metric_diff(healthy_path, dd_path, metric):
    """
    Compute (dd - healthy) / healthy for each method.
    Returns dict of values.
    """

    healthy_df = pd.read_csv(healthy_path)
    dd_df = pd.read_csv(dd_path)

    methods = ["PS", "MI", "LogISI", "CMA", "Unified"]
    prefixes = ["ps", "mi", "logisi", "cma", "unified"]

    cols = [f"{p}_{metric}" for p in prefixes]

    healthy_means = [healthy_df[col].mean() for col in cols]
    dd_means = [dd_df[col].mean() for col in cols]

    diff_dict = {}

    for i, method in enumerate(methods):
        h = healthy_means[i]
        d = dd_means[i]

        diff = (d - h) / h if h != 0 else np.nan
        diff_dict[method] = diff

    return diff_dict

pd_healthy_path = os.path.join('thesis', 'pd_data_frame_healthy.csv')
pd_dd_path = os.path.join('thesis', 'pd_data_frame_dd.csv')
burst_diff = pd_metric_diff(pd_healthy_path, pd_dd_path, "burst_rate")
spikes_diff = pd_metric_diff(pd_healthy_path, pd_dd_path, "%_spikes_in_burst")
all_diffs = {
    "burst_rate": burst_diff,
    "spikes_in_burst": spikes_diff
}

# save to file
path = os.path.join('thesis', 'groups_diffs.txt')

with open(path, "w") as f:
    f.write("Burst Rate Differences:\n")
    for k, v in burst_diff.items():
        f.write(f"{k}: {v:.5f}\n")

    f.write("\n% Spikes in Burst Differences:\n")
    for k, v in spikes_diff.items():
        f.write(f"{k}: {v:.5f}\n")