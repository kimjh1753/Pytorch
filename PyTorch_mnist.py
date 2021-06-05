# 필요 라이브러리 import
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms

# 훈련을 할 때 가용 gpu가 있다면 gpu를 사용하고, 그렇지 않다면 cpu를 사용
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'current device is {device}') 
# current device is cuda

# 데이터 로드 및 처리
train_data = datasets.MNIST(
    root = 'data',
    train = True,
    download = True,
    transform = transforms.ToTensor())

test_data = datasets.MNIST(
    root = 'data',
    train = False,
    download = True,
    transform = transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size = 32
)

test_loader = torch.utils.data.DataLoader(
    test_data, batch_size = 32
)

# for (x_train, y_train) in train_loader:
#     print(f'x_train size : {x_train.size()}, x_train type : {x_train.type()}')
#     print(f'y_train size : {y_train.size()}, x_train type : {y_train.type()}')
# x_train size : torch.Size([32, 1, 28, 28]), x_train type : torch.FloatTensor
# y_train size : torch.Size([32]), x_train type : torch.LongTensor

# modeling
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        x = self.fc3(x)
        x = F.log_softmax(x)
        return x

model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr = 1e-2,
    momentum = 0.5
)

print(model)
# Net(
#   (fc1): Linear(in_features=784, out_features=512, bias=True)
#   (fc2): Linear(in_features=512, out_features=256, bias=True)
#   (fc3): Linear(in_features=256, out_features=10, bias=True)
# )

# train, test
def train(model, train_loader, optimizer, log_interval):
    model.train()

    for batch_idx, (image, label) in enumerate(train_loader):
        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(f'Epochs : {epoch} [{batch_idx * len(image)}/{len(train_loader.dataset)} ({100 * batch_idx / len(train_loader)})%')

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(device)
            label = label.to(device)

            output = model(image)
            prediction = output.max(1, keepdim = True)[1]
            test_loss += criterion(output, label).item() 
            correct += prediction.eq(label.view_as(prediction)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100 * correct / len(test_loader.dataset)
    return test_loss, accuracy

# print loss, accuracy
for epoch in range(1, 11):               
    train(model, train_loader, optimizer, log_interval=200)
    test_loss, accuracy = test(model, test_loader)
    print(f'Test Loss : {test_loss:.4f}, Accuracy : {accuracy}')

# Epochs : 1 [0/60000 (0.0)%
# Epochs : 1 [6400/60000 (10.666666666666666)%
# Epochs : 1 [12800/60000 (21.333333333333332)%
# Epochs : 1 [19200/60000 (32.0)%
# Epochs : 1 [25600/60000 (42.666666666666664)%
# Epochs : 1 [32000/60000 (53.333333333333336)%
# Epochs : 1 [38400/60000 (64.0)%
# Epochs : 1 [44800/60000 (74.66666666666667)%
# Epochs : 1 [51200/60000 (85.33333333333333)%
# Epochs : 1 [57600/60000 (96.0)%
# Test Loss : 0.0698, Accuracy : 42.01
# Epochs : 2 [0/60000 (0.0)%
# Epochs : 2 [6400/60000 (10.666666666666666)%
# Epochs : 2 [12800/60000 (21.333333333333332)%
# Epochs : 2 [19200/60000 (32.0)%
# Epochs : 2 [25600/60000 (42.666666666666664)%
# Epochs : 2 [32000/60000 (53.333333333333336)%
# Epochs : 2 [38400/60000 (64.0)%
# Epochs : 2 [44800/60000 (74.66666666666667)%
# Epochs : 2 [51200/60000 (85.33333333333333)%
# Epochs : 2 [57600/60000 (96.0)%
# Test Loss : 0.0382, Accuracy : 63.86

# ''''''

# Epochs : 9 [0/60000 (0.0)%
# Epochs : 9 [6400/60000 (10.666666666666666)%
# Epochs : 9 [12800/60000 (21.333333333333332)%
# Epochs : 9 [19200/60000 (32.0)%
# Epochs : 9 [25600/60000 (42.666666666666664)%
# Epochs : 9 [32000/60000 (53.333333333333336)%
# Epochs : 9 [38400/60000 (64.0)%
# Epochs : 9 [44800/60000 (74.66666666666667)%
# Epochs : 9 [51200/60000 (85.33333333333333)%
# Epochs : 9 [57600/60000 (96.0)%
# Test Loss : 0.0112, Accuracy : 89.48
# Epochs : 10 [0/60000 (0.0)%
# Epochs : 10 [6400/60000 (10.666666666666666)%
# Epochs : 10 [12800/60000 (21.333333333333332)%
# Epochs : 10 [19200/60000 (32.0)%
# Epochs : 10 [25600/60000 (42.666666666666664)%
# Epochs : 10 [32000/60000 (53.333333333333336)%
# Epochs : 10 [38400/60000 (64.0)%
# Epochs : 10 [44800/60000 (74.66666666666667)%
# Epochs : 10 [51200/60000 (85.33333333333333)%
# Epochs : 10 [57600/60000 (96.0)%
# Test Loss : 0.0108, Accuracy : 89.89