import torch
from tqdm import tqdm

from utils import device
import torch.nn as nn


def train(model, trainloader, valloader, optimizer, scheduler, criterion, epochs=10, input_size=32 * 32 * 3,
          summary_writer=None):
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model.to(device)
    model.train()
    for epoch in range(epochs):
        with tqdm(trainloader) as pbar:
            for i, (images, labels) in enumerate(pbar):
                images = images.view(-1, input_size).to(device)
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels.to(device))
                loss.backward()
                optimizer.step()
                accuracy = (output.argmax(dim=1) == labels.to(device)).float().mean()
                pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item(), lr=optimizer.param_groups[0]['lr'])
                if summary_writer:
                    summary_writer.add_scalar("Loss/train", loss.item(), epoch * len(trainloader) + i)
                    summary_writer.add_scalar("Accuracy/train", accuracy.item(), epoch * len(trainloader) + i)

        # Validation
        model.eval()
        val_loss = 0
        val_accuracy = 0
        with torch.no_grad():
            for images, labels in valloader:
                images = images.view(-1, input_size).to(device)
                output = model(images)
                val_loss += criterion(output, labels.to(device)).item()
                val_accuracy += (
                    (output.argmax(dim=1) == labels.to(device)).float().mean().item()
                )
        val_loss /= len(valloader)
        val_accuracy /= len(valloader)

        # Update learning rate
        scheduler.step()

        if summary_writer:
            summary_writer.add_scalar("Loss/val", val_loss, epoch)
            summary_writer.add_scalar("Accuracy/val", val_accuracy, epoch)

        print(
            f"Epoch {epoch + 1}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}"
        )

    if summary_writer:
        summary_writer.flush()
        summary_writer.close()
