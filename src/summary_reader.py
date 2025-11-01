import pandas as pd
import matplotlib.pyplot as plt
import settings

last_epoch = 42

def plot_summary(last_epoch = None):
    history = pd.read_csv(settings.HISTORY_PATH)
    if last_epoch is not None:
        history = history[:last_epoch]
    
    for metric_idx in range(1, len(history.columns)//2 + 1):
        plot_metric(history, metric_idx)

    build_description(history) 

def build_description(history):
    val_cers = history["val_CER"]
    epoch_saved = val_cers.idxmin()

    saved_model_train_cer = history["CER"][epoch_saved]
    saved_model_val_cer = history["val_CER"][epoch_saved]
    saved_model_train_wer = history["WER"][epoch_saved]
    saved_model_val_wer = history["val_WER"][epoch_saved]
    saved_model_train_loss = history["loss"][epoch_saved]
    saved_model_val_loss = history["val_loss"][epoch_saved]   

    with open(settings.TRAINING_METRICS_FILE_PATH, "w") as file:
        file.write(f"Total epochs: {len(val_cers)} \n")
        file.write(f"Epoch saved: {epoch_saved+1} \n")
        file.write(f"Train Loss: {saved_model_train_loss}\n")
        file.write(f"Validation Loss: {saved_model_val_loss}\n")
        file.write(f"Train CER: {saved_model_train_cer*100: .2f}% \n")
        file.write(f"Validation CER: {saved_model_val_cer*100: .2f}%\n")
        file.write(f"Train WER: {saved_model_train_wer*100: .2f}%\n")
        file.write(f"Validation WER: {saved_model_val_wer*100: .2f}%\n")

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
    
    plt.savefig(settings.PLOTS_PATH + f"plot_{metric_name}.png")
    plt.show()


if __name__ == "__main__":
    plot_summary()