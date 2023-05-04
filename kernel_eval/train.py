"""library for train and test functions"""
from typing import Union
import torch
from torch import optim
from torch import nn
from torch.utils.data import IterableDataset
import pkbar
from tqdm import tqdm


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


def train_model(model: nn.Module, dataloader: IterableDataset, learning_rate: float,
                model_name:str, epochs: int, batch_size: int,
                device: str = "cpu") -> Union[nn.Module, float]:
    """
    Function to train a given model with a given dataset
    
    Parameters:
        model: nn.Module - the model to train
        dataloader: IterableDataset - the dataset to train on
        epochs: int - the number of training epochs
        batch_size: int - the batch size to calculate the progress bar
        device: str - the device to train on (cpu or cuda)
    
    Returns:
        model: nn.Module - the trained model
        train_accuracy: float - the accuracy at the end of the training
    """
    # initialize model, loss function, optimizer and so on
    model = model.to(device)
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)

    for epoch in range(0, epochs):
        # every epoch a new progressbar is created
        # also, depending on the epoch the learning rate gets adjusted before
        # the network is set into training mode
        model.train()
        kbar = pkbar.Kbar(target=624//batch_size, epoch=epoch, num_epochs=epochs,
                          width=20, always_stateful=True)

        correct = 0
        total = 0
        running_loss = 0.0
        # adjust the learning rate
        adjust_learning_rate(optimizer, epoch, epochs, learning_rate)

        for batch_idx, (data, label) in enumerate(dataloader):
            data, label = data.to(device), label.float().to(device)
            optimizer.zero_grad()
            output = model(data)

            loss = loss_fn(output, label)
            loss.backward()

            optimizer.step()
            _, predicted = output.max(-1)

            # calculate the current running loss as well as the total accuracy
            # and update the progressbar accordingly
            running_loss += loss.item()
            total += label.size(0)
            correct += predicted.eq(label).sum().item()

            kbar.update(batch_idx, values=[("loss", running_loss/(batch_idx+1)),
                                           ("acc", 100. * correct / total)])

    return model, 100. * correct / total


def test_model(model: nn.Module, dataloader: IterableDataset,
               batch_size: int, device: str="cpu") -> float:
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
        for _, (data, label) in tqdm(enumerate(dataloader), total=154//batch_size):
            data, label = data.to(device), label.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
        print(f"Test Accuracy: {100. * correct / total}%")

    return 100. * correct / total