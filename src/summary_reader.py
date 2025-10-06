import pandas as pd
import matplotlib.pyplot as plt
import settings

def main():
    history = pd.read_csv(settings.HISTORY_PATH)
    
    for metric_idx in range(1, len(history.columns)//2 + 1):
        plot_metric(history, metric_idx)

def plot_metric(history, metric_idx):
    
    metric_name = history.columns[metric_idx]
    train_metric = history[metric_name]
    val_metric = history[f"val_{metric_name}"]

    plt.figure(figsize=(8, 5))
    plt.plot(train_metric, label=metric_name)
    plt.plot(val_metric, label=f"val_{metric_name}")
    
    plt.title(metric_name)
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(visible=True, linestyle="--", alpha=0.6)

    plt.show()


if __name__ == "__main__":
    main()