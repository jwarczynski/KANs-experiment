from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torchvision


def get_loaders(dataset='mnist', batch_size=64):
    if dataset == 'cifar10':
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

        trainloader = DataLoader(cifar10_trainset, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(cifar10_testset, batch_size=batch_size, shuffle=False)
        valloader = DataLoader(cifar10_validset, batch_size=batch_size, shuffle=False)
    elif dataset == 'mnist':
        mnist_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

        mnist_full_trainset = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=mnist_transform
        )

        mnist_testset = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=mnist_transform
        )

        train_size = int(0.8 * len(mnist_full_trainset))
        valid_size = len(mnist_full_trainset) - train_size
        mnist_trainset, mnist_validset = random_split(mnist_full_trainset, [train_size, valid_size])
        trainloader = DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True)
        valloader = DataLoader(mnist_validset, batch_size=batch_size, shuffle=False)
        testloader = DataLoader(mnist_testset, batch_size=batch_size, shuffle=False)
    else:
        raise ValueError("Dataset not supported")

    return trainloader, valloader, testloader