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

# save to file
path = os.path.join('thesis', 'ttests.txt')
array = np.array(list(results_dict.items()), dtype=object)
np.savetxt(path, array, fmt="%s", delimiter=":")