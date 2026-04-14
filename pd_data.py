from pathlib import Path
import numpy as np
import pandas as pd
import os

source_path = Path("/Users/riyajuneja/thesis/Aristieta_Parker_Gao_Gittis_Rubin_2024_D1_GPe-main/data")
save_path = Path("/Users/riyajuneja/thesis/pd_data")

def save_pd_data():
    save_path.mkdir(exist_ok=True, parents=True)

    init_data = []

    # create dataframe
    col_names = ['actual_rate', 'cv', 'isi_dist', 
                'num_spikes', 'burst_firing_rate', 'avg_ISI_within_bursts', 'burst_rate', '%_spikes_in_burst', '%_time_spent_bursting', 'firing_rate_non_bursting', 'burst_firing_rate_inc']
    df_pd = pd.DataFrame(columns=col_names)

    # iterate through neurons
    for neuron_dir in sorted(source_path.glob("*/pre_processed_data/*/Neuron_*")):
        spikes_file = neuron_dir / 'spikes.txt'
        light_on_file = neuron_dir / 'light_on.txt'

        # read initial data
        with open(spikes_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
        if len(lines) == 0:
            continue
        spikes = np.loadtxt(spikes_file, comments = '#', ndmin = 1)
        light_on = np.loadtxt(light_on_file, comments = '#', ndmin = 1)

        init_data.append({
            "spikes": spikes,
            "light_on": float(light_on[0]),
            "brain_region": neuron_dir.parents[2].name,
            "group": neuron_dir.parent.name,
            "neuron": neuron_dir.name,
        })

    # calculate recording length
    min_light_on = min(data["light_on"] for data in init_data)

    rows = []

    # save spikes within recording length
    for i, data in enumerate(init_data):
        train_dir = save_path / f"train_{i:03d}"
        train_dir.mkdir(exist_ok=True, parents=True)

        start = data["light_on"] - min_light_on
        end = data["light_on"]

        # start at min_light_on and work backwards
        spikes_trunc = data["spikes"][(data["spikes"] >= start) & (data["spikes"] < end)]
        spikes_shifted = spikes_trunc - data["light_on"]
        np.savetxt(train_dir / "spikes.txt", spikes_shifted)

        # save metadata
        all_metadata = {
            "T": min_light_on,
            "brain_region": data["brain_region"],
            "group": data["group"],
            "neuron": data["neuron"]
        }

        metadata_array = np.array(list(all_metadata.items()), dtype=object)
        np.savetxt(train_dir / "metadata.txt", metadata_array, fmt = "%s", delimiter=":")
        rows.append(all_metadata)

    # save dataframe
    df_pd = pd.DataFrame(rows)
    frame_path = os.path.join('thesis', 'pd_data_frame.csv')
    df_pd.to_csv(frame_path, index=False)

# return T, N
def return_params():
    init_data = []
    # iterate through neurons
    for neuron_dir in sorted(source_path.glob("*/pre_processed_data/*/Neuron_*")):
        spikes_file = neuron_dir / 'spikes.txt'
        light_on_file = neuron_dir / 'light_on.txt'

        # read initial data
        with open(spikes_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
        if len(lines) == 0:
            continue
        spikes = np.loadtxt(spikes_file, comments = '#', ndmin = 1)
        light_on = np.loadtxt(light_on_file, comments = '#', ndmin = 1)

        init_data.append({
            "spikes": spikes,
            "light_on": float(light_on[0]),
            "brain_region": neuron_dir.parents[2].name,
            "group": neuron_dir.parent.name,
            "neuron": neuron_dir.name,
        })

    # calculate recording length
    min_light_on = min(data["light_on"] for data in init_data)

    # return params
    return min_light_on, len(init_data)

if __name__ == "__main__":
    save_pd_data()