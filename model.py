"""
Implementation of AlexNet, from paper
"ImageNet Classification with Deep Convolutional Neural Networks" by Alex Krizhevsky et al.

See: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
"""



import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters

EPOCHS = 90
BATCH_SIZE = 128
MOMENTUM = 0.9
LR_DECAY = 0.0005
LR_INIT = 0.01
IMG_DIM = 227
NUM_CLASSES = 1000
DEVICE = device

INPUT_DIR = 'alexnet_data_in'
TRAIN_DIR = 'alexnet_data_in/imagenet'
OUTPUT_DIR = 'alexnet_data_out'
LOG_DIR = 'alexnet_data_out'
CHECKPOINT_DIR = OUTPUT_DIR + '/models'

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# Model

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    

if __name__ == '__main__':
    model = AlexNet(NUM_CLASSES).to(DEVICE)
    print(model)
    wandb.init(project='alexnet', entity='wandb')
    wandb.watch(model)
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LR_INIT, momentum=MOMENTUM)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=LR_DECAY)
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(IMG_DIM),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))
    train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    for epoch in range(EPOCHS):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        lr_scheduler.step()
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('Learning Rate: {}'.format(lr_scheduler.get_lr()))
        print('Loss: {}'.format(loss.item()))
        wandb.log({'Loss': loss.item()})
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f'alexnet_{epoch + 1}.pth'))
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'alexnet.pth'))
    wandb.save(os.path.join(CHECKPOINT_DIR, 'alexnet.pth'))
    wandb.finish()


