"""library for train and test functions"""

import torch
from torch import optim
from torch import nn
from torch.utils.data import IterableDataset
import pkbar
from tqdm import tqdm

def train_model(model: nn.Module, dataloader: IterableDataset,
                epochs: int, device: str = "cpu") -> nn.Module:
    """
    Function to train a given model with a given dataset
    
    Parameters:
        model: nn.Module - the model to train
        dataloader: IterableDataset - the dataset to train on
        epochs: int - the number of training epochs
        device: str - the device to train on (cpu or cuda)
    
    Returns:
        model: nn.Module - the trained model
    """
    # initialize model, loss function, optimizer and so on
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)


    for epoch in range(0, epochs):
        # every epoch a new progressbar is created
        # also, depending on the epoch the learning rate gets adjusted before
        # the network is set into training mode
        model.train()
        kbar = pkbar.Kbar(target=len(dataloader)-1, epoch=epoch, num_epochs=epochs,
                          width=20, always_stateful=True)

        correct = 0
        total = 0
        running_loss = 0.0

        for batch_idx, (data, label) in enumerate(dataloader):
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, label)
            loss.backward()

            optimizer.step()
            _, predicted = output.max(1)

            # calculate the current running loss as well as the total accuracy
            # and update the progressbar accordingly
            running_loss += loss.item()
            total += label.size(0)
            correct += predicted.eq(label).sum().item()

            kbar.update(batch_idx, values=[("loss", running_loss/(batch_idx+1)),
                                           ("acc", 100. * correct / total)])

    return model


def test_model(model: nn.Module, dataloader: IterableDataset, device: str="cpu") -> None:
    """
    Function to test a given model with a given dataset
    
    Parameters:
        model: nn.Module - the model to test
        dataloader: IterableDataset - the dataset to test on
        device: str - the device to test on (cpu or cuda)
    
    Returns:
        None
    """
    # test the model without gradient calculation and in evaluation mode
    with torch.no_grad():
        model = model.to(device)
        model.eval()
        correct = 0
        total = 0
        for _, (data, label) in tqdm(enumerate(dataloader), total=len(dataloader)):
            data, label = data.to(device), label.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
        print(f"Test Accuracy: {100. * correct / total}%")
