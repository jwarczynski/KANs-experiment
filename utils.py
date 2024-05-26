import os
import torch
import torch.nn as nn
import torch.optim as optim

MODEL_DIR = "models"
model_name_template = "{dataset}-{model}-e{epoch}-l{layer_widths}-g{grid_size}-k{spline_order}.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_model(model, model_name):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    torch.save(model.state_dict(), f"{MODEL_DIR}/{model_name}")


def test_accuracy(model, testloader, input_size=32 * 32 * 3, writer=None):
    model.eval()
    test_acc = 0
    with torch.no_grad():
        for images, labels in testloader:
            images = images.view(-1, input_size).to(device)
            output = model(images)
            test_acc += (
                (output.argmax(dim=1) == labels.to(device)).float().mean().item()
            )
    test_acc /= len(testloader)
    if writer:
        writer.add_scalar("Accuracy/test", test_acc)
    return test_acc


def get_optimizer_scheduler_criterion(model, lr=1e-3, weight_decay=1e-4):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    criterion = nn.CrossEntropyLoss()
    return optimizer, scheduler, criterion


def get_model_name(layer_widths, grid_size, spline_order, model='kan', epoch=10, dataset='cifar10'):
    return model_name_template.format(
        dataset=dataset,
        model=model,
        epoch=epoch,
        layer_widths='_'.join(map(str, layer_widths)),
        grid_size=grid_size,
        spline_order=spline_order
    )