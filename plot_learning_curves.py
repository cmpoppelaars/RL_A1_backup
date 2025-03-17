import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

def load_data(models, learning_rates, epsilons, network_sizes, update_to_data_ratios, update_frequencies=None, buffer_sizes=None):
    
    data_dict = {}
    for model in models:
        data_dict[model] = {
            "lr": {lr: [] for lr in learning_rates},
            "eps": {eps: [] for eps in epsilons},
            "nwsize": {nw: [] for nw in network_sizes},
            "ur": {ur: [] for ur in update_to_data_ratios},
        }
        
        if "ER" in model:
            data_dict[model]["buffer_size"] = {buffer_size: [] for buffer_size in buffer_sizes}
        if "TN" in model:
            data_dict[model]["update_freq"] = {update_freq: [] for update_freq in update_frequencies}
    
    NDQN_count = 0
    DQN_ER_count = 0
    DQN_TN_count = 0
    DQN_ER_TN_count = 0

    er_true = False
    tn_true = False
    for model in models:
        # Check if we need extra for loops
        if "ER" in model:
            er_true = True

        if "TN" in model:
            tn_true = True

        for learning_rate in learning_rates:
            for epsilon in epsilons:
                for network_size in network_sizes:
                    for update_ratio in update_to_data_ratios:

                        if er_true:
                            if tn_true:
                                # DQN w ER and TN
                                for buffer_size in buffer_sizes:
                                     for update_freq in update_frequencies:
                                        dir_name = f"{model}_data"
                                        filename = os.path.join(dir_name, f"{model}_buffer_size{buffer_size}_update_freq{update_freq}_data_update_ratio{update_ratio}_lr{learning_rate}_eps{epsilon}_nwsize{network_size}.npz")
                                        if os.path.exists(filename):
                                            DQN_ER_TN_count += 1
                                            data = np.load(filename)
                                            curve, timesteps = data["learning_curve"], data["timesteps"]
                                            data_dict[model]["lr"][learning_rate].append((timesteps, curve))
                                            data_dict[model]["eps"][epsilon].append((timesteps, curve))
                                            data_dict[model]["nwsize"][network_size].append((timesteps, curve))
                                            data_dict[model]["ur"][update_ratio].append((timesteps, curve))
                                            data_dict[model]["update_freq"][update_freq].append((timesteps, curve))
                                            data_dict[model]["buffer_size"][buffer_size].append((timesteps, curve))

                            else:
                                # DQN w ER
                                for buffer_size in buffer_sizes:
                                    dir_name = f"{model}_data"
                                    filename = os.path.join(dir_name, f"{model}_buffersize{buffer_size}_data_update_ratio{update_ratio}_lr{learning_rate}_eps{epsilon}_nwsize{network_size}.npz")
                                    if os.path.exists(filename):
                                        DQN_ER_count += 1
                                        data = np.load(filename)
                                        curve, timesteps = data["learning_curve"], data["timesteps"]
                                        data_dict[model]["lr"][learning_rate].append((timesteps, curve))
                                        data_dict[model]["eps"][epsilon].append((timesteps, curve))
                                        data_dict[model]["nwsize"][network_size].append((timesteps, curve))
                                        data_dict[model]["ur"][update_ratio].append((timesteps, curve))
                                        data_dict[model]["buffer_size"][buffer_size].append((timesteps, curve))

                        if tn_true:
                            # DWN w TN
                            for update_freq in update_frequencies:
                                dir_name = f"{model}_data"
                                filename = os.path.join(dir_name, f"{model}_update_freq{update_freq}_data_update_ratio{update_ratio}_lr{learning_rate}_eps{epsilon}_nwsize{network_size}.npz")
                                if os.path.exists(filename):
                                    DQN_TN_count += 1
                                    data = np.load(filename)
                                    curve, timesteps = data["learning_curve"], data["timesteps"]
                                    data_dict[model]["lr"][learning_rate].append((timesteps, curve))
                                    data_dict[model]["eps"][epsilon].append((timesteps, curve))
                                    data_dict[model]["nwsize"][network_size].append((timesteps, curve))
                                    data_dict[model]["ur"][update_ratio].append((timesteps, curve))
                                    data_dict[model]["update_freq"][update_freq].append((timesteps, curve))
                        else:
                            # NDQN        
                            dir_name = f"{model}_data"  # Use model as directory name
                            filename = os.path.join(
                                dir_name,
                                f"{model}_data_update_ratio{update_ratio}_lr{learning_rate}_eps{epsilon}_nwsize{network_size}.npz"
                            )

                            if os.path.exists(filename):
                                NDQN_count+=1
                                data = np.load(filename)
                                curve, timesteps = data["learning_curve"], data["timesteps"]
                                data_dict[model]["lr"][learning_rate].append((timesteps, curve))
                                data_dict[model]["eps"][epsilon].append((timesteps, curve))
                                data_dict[model]["nwsize"][network_size].append((timesteps, curve))
                                data_dict[model]["ur"][update_ratio].append((timesteps, curve))

    print(f"Loaded {NDQN_count} NDQN files")
    print(f"Loaded {DQN_ER_TN_count} DQN ER TN files")
    print(f"Loaded {DQN_ER_count} DQN ER files")
    print(f"Loaded {DQN_TN_count} DQN TN files")
    return data_dict

def plot_mean_learning_curves(data_dict, models):
    hyperparams = ["lr", "eps", "nwsize", "ur", "buffer_size", "update_freq"]
    titles = {
        "lr": "Learning Rate",
        "eps": "Epsilon",
        "nwsize": "Network Size",
        "ur": "Update Ratio",
        "buffer_size": "Buffer size",
        "update_freq": "TN update freq"
    }
    
    # Define a color for each subplot (different hyperparameter → different color family)
    hyperparam_colors = {
        "lr": cm.Reds,          # Red for Learning Rate
        "eps": cm.Greys,        # Black/Grey for Epsilon
        "nwsize": cm.Blues,     # Blue for Network Size
        "ur": cm.Purples,       # Purple for Update Ratio
        "buffer_size": cm.Greens,   # Green for Buffer Size
        "update_freq": cm.Oranges,  # Orange for TN update freq
    }
    
    # Define different line styles for distinguishing models
    model_linestyles = {
        "NDQN": "solid",
        "DQN_ER": "dotted",
        "DQN_TN": "dashed",
        "DQN_ER_TN": "dashdot"
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for ax, key in zip(axes, hyperparams):
        plotted_values = set()  # Set to keep track of plotted values for legend
        
        for model in models:

            # Skip 'update_freq' if the model does not contain 'TN'
            if key == "update_freq" and "TN" not in model:
                continue
            # Skip 'buffer_size' if the model does not contain 'ER'
            if key == "buffer_size" and "ER" not in model:
                continue
            
            param_values = sorted(data_dict[model][key].keys())
            num_curves = len(param_values)
            
            colormap = hyperparam_colors[key]  # Assign color based on hyperparameter
            color_range = np.linspace(0.2, 0.9, num_curves)  # Avoid extreme white colors
            
            for i, value in enumerate(param_values):
                curves = data_dict[model][key][value]
                if curves:
                    max_timesteps = max(len(t[1]) for t in curves)
                    mean_curve = np.zeros(max_timesteps)
                    count = np.zeros(max_timesteps)
                    
                    for timesteps, curve in curves:
                        length = len(curve)
                        mean_curve[:length] += curve
                        count[:length] += 1
                    
                    mean_curve[count > 0] /= count[count > 0]
                    
                    # Select color from the adjusted range
                    color = colormap(color_range[i])
                    linestyle = model_linestyles[model]  # Assign model-specific linestyle
                    
                    # Plot the curve for all models (but keep track of unique values)
                    ax.plot(timesteps, mean_curve, color=color, linestyle=linestyle)
                    
                    # Only add unique hyperparameter values to the legend
                    if value not in plotted_values:
                        ax.plot([], [], label=f"{key}: {value}", color=color, linestyle="solid")  # Add empty plot for the legend
                        plotted_values.add(value)  # Add value to the set to avoid re-plotting in the legend
                    
        ax.set_title(f"Effect of {titles[key]}")
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Average Reward")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig("learning_curves_hyperparameters.pdf", format="pdf", dpi=400)

def plot_best_agents():
    models = ["NDQN", "DQN_ER", "DQN_TN", "DQN_ER_TN"]
    
    data_dict = load_data(
        models,
        [0.001],
        [0.5],
        [128],
        [0.1],
        [5000],
        [75000]
    )

    plt.figure()
    for model in data_dict:
        time_steps = data_dict[model]['lr'][0.001][0][0]
        rewards = data_dict[model]['lr'][0.001][0][1]
        plt.plot(time_steps, rewards, label=model)

    plt.xlabel("Timesteps")
    plt.ylabel("Average reward")
    plt.legend()
    plt.grid()
    plt.savefig('best_models.pdf', dpi=400)

def plot_hyperparameter_aggregated_curves():
    models = ["NDQN", "DQN_ER", "DQN_TN", "DQN_ER_TN"]
    
    data_dict = load_data(
        models,
        [0.1, 0.001, 1e-5],
        [0.05, 0.25, 0.5],
        [32, 64, 128],
        [0.1, 0.5, 1.0],
        [5000, 25000, 50000],
        [25000, 75000, 125000]
    )
    plot_mean_learning_curves(data_dict, models)

if __name__ == '__main__':
    plot_hyperparameter_aggregated_curves()
    plot_best_agents()