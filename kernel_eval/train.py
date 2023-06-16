"""library for train and test functions"""
from typing import Union, List
import torch
from torch import optim
from torch import nn
from torch.utils.data import IterableDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pkbar
from tqdm import tqdm
from kernel_eval.utils import save_model
import wandb


def adjust_learning_rate(optimizer, epoch: int, epochs: int, learning_rate: int) -> None:
    """
    helper function to adjust the learning rate
    according to the current epoch to prevent overfitting.
    
    Parameters:
        optimizer: the optimizer to adjust the learning rate with
        epoch: the current epoch
        epochs: the total number of epochs
        learning_rate: the learning rate to adjust

    Returns:
        None
    """
    new_lr = learning_rate
    if epoch >= torch.floor(torch.Tensor([epochs*0.5])):
        new_lr /= 10
    if epoch >= torch.floor(torch.Tensor([epochs*0.75])):
        new_lr /= 10
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def train_model(model: nn.Module, train_dataloader: IterableDataset,
                validation_dataloader: IterableDataset,
                learning_rate: float, epochs: int, batch_size: int,
                device: str = "cpu", model_type: str = "no_model", depthwise: bool = True,
                mpath_out: str = "./models/") -> Union[nn.Module, float, List[float], List[float]]:
    """
    Function to train a given model with a given dataset
    
    Parameters:
        model: nn.Module - the model to train
        train_dataloader: IterableDataset - the dataset to train on
        learning_rate: float - the learning rate of SGD
        epochs: int - the number of training epochs
        batch_size: int - the batch size to calculate the progress bar
        device: str - the device to train on (cpu or cuda)
        model_type: str - the name of model (VGG, ResNet)
        depthwise: bool - if true enables depthwise conv
        mpath_out: str - saving path for the model with the best train_acc
    
    Returns:
        model: nn.Module - the trained model
        train_accuracy: float - the accuracy at the end of the training
        train_losses: List[float] - the losses at the end of each epoch
        train_accs: List[float] - the accuracies at the end of each epoch
    """

    wandb.init(
        # set the wandb project where this run will be logged
        project="kernel_optimization",

        # track hyperparameters and run metadata
        config={
        "learning_rate": str(learning_rate),
        "architecture": model_type,
        "dataset": "bioimages",
        "epochs": str(epochs),
        "depthwise": depthwise
        }
    )

    # initialize model, loss function, optimizer and so on
    train_accs: List[float] = []
    train_losses: List[float] = []

    best_validation_acc: float = 0.0
    best_model: torch.Module = model.to(device)
    best_epoch: int = 0

    model = model.to(device)
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=5, verbose=True)


    for epoch in range(0, epochs):
        # every epoch a new progressbar is created
        # also, depending on the epoch the learning rate gets adjusted before
        # the network is set into training mode

        tp_val: float = 0
        fp_val: float = 0
        fn_val: float = 0
        model.train()
        kbar = pkbar.Kbar(target=len(train_dataloader) - 1, epoch=epoch, num_epochs=epochs,
                          width=20, always_stateful=True)

        correct: int = 0
        total: int = 0
        running_loss: float = 0.0
        epoch_loss: List[float] = []
        epoch_acc: List[float] = []
        # adjust the learning rate
        # adjust_learning_rate(optimizer, epoch, epochs, learning_rate)

        for batch_idx, (data, label) in enumerate(train_dataloader):
            data, label = data.to(device), label.float().to(device)

            optimizer.zero_grad()
            output = model(data)
            # squeeze the output to fit the label shape
            output = output.squeeze(-1)

            loss = loss_fn(output, label)
            loss.backward()

            optimizer.step()
            predicted = output.round()

            # calculate the current running loss as well as the total accuracy
            # and update the progressbar accordingly
            running_loss += loss.item()
            epoch_loss.append(loss.item())
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
            epoch_acc.append(100. * correct / total)

            # Calculate metrics
            # Update tp_val, fp_val, and fn_val counters
            tp_val += ((predicted == 1) & (label == 1)).sum().item()
            fp_val += ((predicted == 1) & (label == 0)).sum().item()
            fn_val += ((predicted == 0) & (label == 1)).sum().item()

            kbar.update(batch_idx, values=[("loss", running_loss/(batch_idx+1)),
                                           ("acc", 100. * correct / total)])
            wandb.log({"train_acc": 100 * correct/total, "train_loss": running_loss/(batch_idx+1)})

        # After the training loop for each epoch
        avg_epoch_loss = sum(epoch_loss) / len(epoch_loss)
        scheduler.step(avg_epoch_loss)

        # append the accuracy and loss of the current epoch to the lists
        train_accs.append(sum(epoch_acc) / len(epoch_acc))
        train_losses.append(sum(epoch_loss) / len(epoch_loss))

        # Calculate average metrics for the epoch
        # Calculate precision, recall, and F1 score
        precision = tp_val / (tp_val + fp_val) if tp_val + fp_val > 0 else 0
        recall = tp_val / (tp_val + fn_val) if tp_val + fn_val > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

        wandb.log({"train_recall": recall, "train_precision": precision, "train_f1": f1_score})


        # for every epoch use the validation set to check if this is the best model yet
        validation_acc = test_model(model, validation_dataloader, device)

        if validation_acc > best_validation_acc:
            best_validation_acc = validation_acc
            best_model = model
            best_epoch = epoch
            save_model(mpath_out, model_type, depthwise, batch_size, learning_rate, epochs, model)

    return best_model, train_accs[best_epoch], train_accs, train_losses


def test_model(model: nn.Module, dataloader: IterableDataset, device: str="cpu") -> float:
    """
    Function to test a given model with a given dataset
    
    Parameters:
        model: nn.Module - the model to test
        dataloader: IterableDataset - the dataset to test on
        device: str - the device to test on (cpu or cuda)
    
    Returns:
        test_accuracy: float - the test accuracy
    """
    # test the model without gradient calculation and in evaluation mode


    with torch.no_grad():
        model = model.to(device)
        model.eval()
        correct = 0
        total = 0
        for _, (data, label) in tqdm(enumerate(dataloader), total=len(dataloader)-1):
            data, label = data.to(device), label.float().to(device)

            output = model(data)
            # squeeze the output to fit the label shape
            output = output.squeeze(-1)
            predicted = output.round()
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
            # wandb.log({"test_acc": 100 * correct / total})
        print(f"Test Accuracy: {100. * correct / total}%")

    return 100. * correct / total
