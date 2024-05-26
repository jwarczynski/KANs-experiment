from typing import Dict, Any

import torchvision
from efficient_kan import KAN
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# Train on MNIST

from train import train
from utils import *


print(f'device: {device}')
RUNS_DIR = 'runs'

def run_experiment(
        layers, model_args: Dict[str, Any],trainloader: DataLoader,
        valloader: DataLoader, testloader: DataLoader, epochs=10,
        dataset='cifar10', model_arch='kan'
):
    model_name = get_model_name(layers, **model_args, model=model_arch, dataset=dataset, epoch=epochs)
    writer = SummaryWriter(f'{RUNS_DIR}/{model_name}')

    model = KAN(layers, **model_args)
    osc = get_optimizer_scheduler_criterion(model)

    train(model, trainloader, valloader, *osc, epochs=epochs, input_size=layers[0], summary_writer=writer)
    test_acc = test_accuracy(model, testloader, input_size=layers[0], writer=writer)
    print(f"Test Accuracy: {test_acc}")

    writer.flush()
    writer.close()

    save_model(model, model_name)
    del model


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Load cifar10
    cifar10_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    cifar10_full_trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=cifar10_transform
    )
    train_size = int(0.8 * len(cifar10_full_trainset))
    valid_size = len(cifar10_full_trainset) - train_size
    cifar10_trainset, cifar10_validset = random_split(cifar10_full_trainset, [train_size, valid_size])

    cifar10_testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=cifar10_transform
    )

    trainloader = DataLoader(cifar10_trainset, batch_size=64, shuffle=True)
    testloader = DataLoader(cifar10_testset, batch_size=64, shuffle=False)
    valloader = DataLoader(cifar10_validset, batch_size=64, shuffle=False)

    layers = [32 * 32 * 3, 128, 64, 32, 10]
    model_args_list = []

    # Generate the first 10 configurations: spline order = 3, grid size doubling from 10
    initial_grid_size = 10
    spline_order_fixed = 3
    for i in range(10):
        model_args_list.append({
            'grid_size': initial_grid_size * (2 ** i),
            'spline_order': spline_order_fixed
        })

    # Generate the next 10 configurations: grid size = 5, spline orders from 2 to 12
    grid_size_fixed = 5
    for spline_order in range(2, 13):
        model_args_list.append({
            'grid_size': grid_size_fixed,
            'spline_order': spline_order
        })

    print(f'args_list: {model_args_list}')

    # Iterate over the model_args_list and run experiments
    for model_args in model_args_list[2:]:
        print(f"Running experiment with model_args: {model_args}")
        run_experiment(layers, model_args, trainloader, valloader, testloader)
        print(f"Finished experiment with model_args: {model_args}\n")
