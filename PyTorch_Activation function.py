import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'current device is {device}')

train_data = datasets.MNIST(
    root = 'data',
    train = True,
    download = False,
    transform = transforms.ToTensor()
)

test_data = datasets.MNIST(
    root = 'data',
    train = False,
    downloaer = False,
    transform = transforms.ToTensor()
)

train_loader = torch.util.data.DataLoader(
    train_data, batch_size = 32
)

test_loader = torch.util.data.DataLoader(
    test_data, batch_size = 32
)

class Net(nn.Moudle):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 512)

    def foward(self, w):
        