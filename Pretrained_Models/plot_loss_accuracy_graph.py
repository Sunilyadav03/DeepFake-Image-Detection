import matplotlib.pyplot as plt

def plot_history(history, model_name):
    """
    Plots the training and validation accuracy and loss from the training history.

    Parameters:
        history (keras.callbacks.History): The history object returned by model.fit().
        model_name (str): The name of the model (used in the plot title).
    """
    # Determine the correct keys for accuracy and loss
    acc_key = 'accuracy' if 'accuracy' in history.history else 'acc'
    val_acc_key = 'val_accuracy' if 'val_accuracy' in history.history else 'val_acc'
    loss_key = 'loss'
    val_loss_key = 'val_loss'

    # Create a figure with two subplots
    plt.figure(figsize=(12, 5))

    # Plot Training and Validation Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history[acc_key], 'bo-', label='Training Accuracy')
    plt.plot(history.history[val_acc_key], 'ro-', label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)  # Add grid for better readability

    # Plot Training and Validation Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history[loss_key], 'bo-', label='Training Loss')
    plt.plot(history.history[val_loss_key], 'ro-', label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)  # Add grid for better readability

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()
 
