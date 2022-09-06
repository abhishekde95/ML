import torch
import numpy as np
import matplotlib.pyplot as plt


# Calculate accuracy (a classification metric)
def accuracy_fn(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.float:
    """Compute the prediction accuracy
    Parameters:
        y_true <torch.Tensor>: True labels
        y_pred <torch.Tensor>: Predicted labels
    Returns:
        acc <torch.float>: Accuracy
    """
    correct = (
        torch.eq(y_true, y_pred).sum().item()
    )  # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_true)) * 100
    return acc


def plot_decision_boundary(
    model: torch.nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    device: torch.device = "cpu",
):
    """Plots decision boundaries of model predicting on X in comparison to y.
    Source - https://madewithml.com/courses/foundations/neural-networks/ (with modifications)
    Parameters:
        model <torch.nn.Module>: Model
        X <torch.Tensor>: Inputs
        Y <torch.Tensor>: Labels
        device <str>: Device where the data and model should be mounted to
    """
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to(device)
    X, y = X.to(device), y.to(device)

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


# Plot linear data or training and test and predictions (optional)
def plot_predictions(
    train_data: torch.Tensor,
    train_labels: torch.Tensor,
    test_data: torch.Tensor,
    test_labels: torch.Tensor,
    predictions: torch.Tensor = None,
):
    """
    Plots linear training data and test data and compares predictions.
    Parameters:
        train_data <torch.Tensor>: Training data
        train_labels <torch.Tensor>: Testing data
        test_data <torch.Tensor>: Testing data
        test_labels <torch.Tensor>: Testing data
        predictions <torch.Tensor>: Predictions
    """
    plt.figure(figsize=(10, 7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        # Plot the predictions in red (predictions were made on the test data)
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Show the legend
    plt.legend(prop={"size": 14})


def print_train_time(start: float, end: float, device: torch.device = None) -> float:
    """Prints difference between start and end time.

    Parameters:
        start (float): Start time of computation (preferred in timeit format).
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time


def train_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device = "cpu",
) -> dict:
    """Returns a dictionary containing the results of model predicting on data_loader.

    Args:
        model (torch.nn.Module): A PyTorch model capable of making predictions on data_loader.
        data_loader (torch.utils.data.DataLoader): The target dataset to predict on.
        loss_fn (torch.nn.Module): The loss function of model.
        accuracy_fn: An accuracy function to compare the models predictions to the truth labels.

    Returns:
        (dict): Results of model making predictions on data_loader.
    """
    train_loss, train_acc = 0, 0
    model.train()
    for i, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss
        acc = accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))
        train_acc += acc

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Scale loss and acc to find the average loss/acc per batch
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)

    return {
        "model_name": model.__class__.__name__,
        "model_loss": train_loss.item(),
        "model_acc": train_acc,
    }


def test_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device = "cpu",
) -> dict:
    """Returns a dictionary containing the results of model predicting on data_loader.

    Args:
        model (torch.nn.Module): A PyTorch model capable of making predictions on data_loader.
        data_loader (torch.utils.data.DataLoader): The target dataset to predict on.
        loss_fn (torch.nn.Module): The loss function of model.
        accuracy_fn: An accuracy function to compare the models predictions to the truth labels.

    Returns:
        (dict): Results of model making predictions on data_loader.
    """
    test_loss, test_acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for i, (X, y) in enumerate(data_loader):
            # Make predictions with the model
            X, y = X.to(device), y.to(device)
            y_pred = model(X)

            # Accumulate the loss and accuracy values per batch
            loss = loss_fn(y_pred, y)
            test_loss += loss
            acc = accuracy_fn(
                y_true=y, y_pred=y_pred.argmax(dim=1)
            )  # For accuracy, need the prediction labels (
            # logits -> pred_prob -> pred_labels)
            test_acc += acc

        # Scale loss and acc to find the average loss/acc per batch
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)

    return {
        "model_name": model.__class__.__name__,
        "model_loss": test_loss.item(),
        "model_acc": test_acc,
    }


def get_model_prediction(
    model: torch.nn.Module, input_data: torch.Tensor
) -> torch.Tensor:
    """Return the prediction of the model.
    Parameters:
        model <torch.nn.Module>: Model
        input_data <torch.Tensor>: input data to the model
    Returns:
        y_pred <torch.Tensor>: Predicted label
    """
    model.eval()
    with torch.inference_mode():
        y_pred = model(input_data)

    return y_pred.argmax(dim=1)
