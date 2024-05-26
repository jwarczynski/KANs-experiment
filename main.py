from typing import Dict, Any

from efficient_kan import KAN
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from train import train
from utils import *
from loaders import get_loaders
import os

print(f'device: {device}')
RUNS_DIR = 'runs'


def run_experiment(
        layers, model_args: Dict[str, Any], trainloader: DataLoader,
        valloader: DataLoader, testloader: DataLoader, epochs=10,
        dataset='cifar10', model_arch='kan'
):
    model_name = get_model_name(layers, **model_args, model=model_arch, dataset=dataset, epoch=epochs)
    if os.path.exists(f'{MODEL_DIR}/{model_name}') or os.path.exists(f'{MODEL_DIR}/{model_name}.pth'):
        print(f"Model {model_name} already exists. Skipping...")
        return

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


if __name__ == '__main__':
    # Load MNIST data
    trainloader, valloader, testloader = get_loaders('mnist', batch_size=128)

    layers = [28*28, 128, 64, 32, 10]
    model_args_list = []

    # Generate the first 9 configurations: spline order = 3, grid size doubling from 5
    initial_grid_size = 5
    spline_order_fixed = 3
    for i in range(9):
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
    for model_args in model_args_list:
        print(f"Running experiment with model_args: {model_args}")
        run_experiment(layers, model_args, trainloader, valloader, testloader, dataset='mnist')
        print(f"Finished experiment with model_args: {model_args}\n")
