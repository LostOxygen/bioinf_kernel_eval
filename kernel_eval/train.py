"""library for train and test functions"""
from typing import Union, List
import torch
from torch import optim
from torch import nn
from torch.utils.data import IterableDataset
import pkbar
from tqdm import tqdm

from .utils import augment_images, normalize_spectral_data


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


def train_model(model: nn.Module, dataloader: IterableDataset,
                learning_rate: float, epochs: int,
                device: str = "cpu") -> Union[nn.Module, float, List[float], List[float]]:
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
        train_losses: List[float] - the losses at the end of each epoch
        train_accs: List[float] - the accuracies at the end of each epoch
    """
    # initialize model, loss function, optimizer and so on
    train_accs = []
    train_losses = []

    model = model.to(device)
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)

    for epoch in range(0, epochs):
        # every epoch a new progressbar is created
        # also, depending on the epoch the learning rate gets adjusted before
        # the network is set into training mode
        model.train()
        kbar = pkbar.Kbar(target=len(dataloader), epoch=epoch, num_epochs=epochs,
                          width=20, always_stateful=True)

        correct = 0
        total = 0
        running_loss = 0.0
        epoch_loss = []
        epoch_acc = []
        # adjust the learning rate
        adjust_learning_rate(optimizer, epoch, epochs, learning_rate)

        for batch_idx, (data, label) in enumerate(dataloader):
            label = label.unsqueeze(0).float().to(device)

            # find amidi-band by searching for the highest mean pixel value over all channels
            mean_pixel_value_every_dimension = torch.mean(data, (1, 2))
            max_wavenumber = torch.argmax(mean_pixel_value_every_dimension)

            # normalize the data
            data = normalize_spectral_data(data, num_channel=data.shape[1],
                                           max_wavenumber=max_wavenumber)

            # apply augmentation to the images
            data = augment_images(data, size=224).to(device)

            optimizer.zero_grad()
            output = model(data)

            loss = loss_fn(output, label)
            loss.backward()

            optimizer.step()
            _, predicted = output.max(-1)

            # calculate the current running loss as well as the total accuracy
            # and update the progressbar accordingly
            running_loss += loss.item()
            epoch_loss.append(loss.item())
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
            epoch_acc.append(100. * correct / total)

            kbar.update(batch_idx, values=[("loss", running_loss/(batch_idx+1)),
                                           ("acc", 100. * correct / total)])
        # append the accuracy and loss of the current epoch to the lists
        train_accs.append(sum(epoch_acc) / len(epoch_acc))
        train_losses.append(sum(epoch_loss) / len(epoch_loss))

    return model, 100. * correct / total, train_accs, train_losses


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
        for _, (data, label) in tqdm(enumerate(dataloader), total=len(dataloader)):
            label = label.to(device)
            # crop the images to the correct size
            data = augment_images(data, size=224).to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
        print(f"Test Accuracy: {100. * correct / total}%")

    return 100. * correct / total
