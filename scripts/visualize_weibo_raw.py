import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Visualize raw EEG data")
    parser.add_argument("--dataset", type=str, default="weibo2014", help="Dataset name (e.g. weibo2014, cho2017, real_physionet)")
    parser.add_argument("--subject", type=int, default=1, help="Subject ID to visualize")
    parser.add_argument("--trial", type=int, default=0, help="Trial index to visualize")
    parser.add_argument("--channels", type=int, default=10, help="Number of channels to plot")
    args = parser.parse_args()

    data_path = Path(f"data/{args.dataset}/processed/subject_{args.subject}.npz")
    if not data_path.exists():
        data_path = Path(f"data/{args.dataset}/subject_{args.subject}.npz")

    if not data_path.exists():
        print(f"Error: Data file not found for {args.dataset} at subject {args.subject}")
        return

    # Load the preprocessed (but not yet denoised) dataset
    print(f"Loading data from {data_path}...")
    data = np.load(data_path, allow_pickle=True)
    X = data["X"]
    sfreq = float(data["sfreq"])
    ch_names = data["ch_names"]

    print(f"Data shape: {X.shape} (trials, channels, timepoints)")
    print(f"Sampling frequency: {sfreq} Hz")

    # Select the specific trial to plot
    trial_data = X[args.trial]
    n_times = trial_data.shape[1]
    times = np.arange(n_times) / sfreq

    # Setup the plot
    n_plot_chans = min(args.channels, len(ch_names))
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate an appropriate offset based on the variance of the data
    # Weibo data has massive variance, so we need a large offset to prevent overlapping
    max_ptp = np.max(np.ptp(trial_data[:n_plot_chans], axis=1))
    offset_step = max_ptp * 1.5 if max_ptp > 0 else 100
    
    print(f"Plotting trial {args.trial}, first {n_plot_chans} channels. Offset step: {offset_step:.2f}")

    offsets = []
    for i in range(n_plot_chans):
        offset = i * offset_step
        offsets.append(offset)
        # Plot each channel with a vertical offset
        ax.plot(times, trial_data[i] + offset, label=ch_names[i], linewidth=1.0)

    # Styling the plot
    ax.set_yticks(offsets)
    ax.set_yticklabels(ch_names[:n_plot_chans])
    ax.set_xlabel("Time (seconds)")
    ax.set_title(f"{args.dataset} Raw Data - Subject {args.subject}, Trial {args.trial} (First {n_plot_chans} channels)")
    ax.grid(True, alpha=0.3)
    
    # Save the plot
    out_dir = Path("results/visualizations")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.dataset}_raw_subj{args.subject}_trial{args.trial}.png"
    
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    print(f"\nSaved visualization to: {out_file}")
    
    # Calculate some basic statistics to show the noise level
    print("\n--- Basic Statistics for this Trial ---")
    print(f"Mean variance across all channels: {np.mean(np.var(trial_data, axis=1)):.2f}")
    print(f"Max amplitude observed: {np.max(trial_data):.2f}")
    print(f"Min amplitude observed: {np.min(trial_data):.2f}")

if __name__ == "__main__":
    main()
