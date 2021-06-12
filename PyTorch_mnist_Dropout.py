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
        self.dropout_prob = 0.5

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = F.dropout(x, training = self.training, p = self.dropout_prob)
        x = self.fc2(x)
        x = F.sigmoid(x)
        x = F.dropout(x, training = self.training, p = self.dropout_prob)
        x = self.fc3(x)
        x = F.log_softmax(x)
        return x

model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr = 1e-2,
    momentum = 0.3
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
            print(f'Epochs : {epoch} [{batch_idx * len(image)}/{len(train_loader.dataset)} ({100 * batch_idx / len(train_loader)}) %, Loss : {loss.item()}')

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
for epoch in range(1, 31):               
    train(model, train_loader, optimizer, log_interval=200)
    test_loss, accuracy = test(model, test_loader)
    print(f'Test Loss : {test_loss:.4f}, Accuracy : {accuracy}')

# Epochs : 1 [0/60000 (0.0) %, Loss : 2.3379757404327393
# Epochs : 1 [6400/60000 (10.666666666666666) %, Loss : 2.3226606845855713
# Epochs : 1 [12800/60000 (21.333333333333332) %, Loss : 2.3034965991973877
# Epochs : 1 [19200/60000 (32.0) %, Loss : 2.2320594787597656
# Epochs : 1 [25600/60000 (42.666666666666664) %, Loss : 2.327909231185913
# Epochs : 1 [32000/60000 (53.333333333333336) %, Loss : 2.333728075027466
# Epochs : 1 [38400/60000 (64.0) %, Loss : 2.248807907104492
# Epochs : 1 [44800/60000 (74.66666666666667) %, Loss : 2.3303139209747314
# Epochs : 1 [51200/60000 (85.33333333333333) %, Loss : 2.317605972290039
# Epochs : 1 [57600/60000 (96.0) %, Loss : 2.2377634048461914
# Test Loss : 0.0712, Accuracy : 29.19
# Epochs : 2 [0/60000 (0.0) %, Loss : 2.30676007270813
# Epochs : 2 [6400/60000 (10.666666666666666) %, Loss : 2.3238329887390137
# Epochs : 2 [12800/60000 (21.333333333333332) %, Loss : 2.2888782024383545
# Epochs : 2 [19200/60000 (32.0) %, Loss : 2.2965235710144043
# Epochs : 2 [25600/60000 (42.666666666666664) %, Loss : 2.2643167972564697
# Epochs : 2 [32000/60000 (53.333333333333336) %, Loss : 2.2213897705078125
# Epochs : 2 [38400/60000 (64.0) %, Loss : 2.243220567703247
# Epochs : 2 [44800/60000 (74.66666666666667) %, Loss : 2.1984810829162598
# Epochs : 2 [51200/60000 (85.33333333333333) %, Loss : 2.153719902038574
# Epochs : 2 [57600/60000 (96.0) %, Loss : 2.1056840419769287
# Test Loss : 0.0630, Accuracy : 45.56

#''''''

# Epochs : 29 [0/60000 (0.0) %, Loss : 0.2107924222946167
# Epochs : 29 [6400/60000 (10.666666666666666) %, Loss : 0.5197393298149109
# Epochs : 29 [12800/60000 (21.333333333333332) %, Loss : 0.35938283801078796
# Epochs : 29 [19200/60000 (32.0) %, Loss : 0.31828397512435913
# Epochs : 29 [25600/60000 (42.666666666666664) %, Loss : 0.4262486696243286
# Epochs : 29 [32000/60000 (53.333333333333336) %, Loss : 0.43041929602622986
# Epochs : 29 [38400/60000 (64.0) %, Loss : 0.2586546242237091
# Epochs : 29 [44800/60000 (74.66666666666667) %, Loss : 0.3686130940914154
# Epochs : 29 [51200/60000 (85.33333333333333) %, Loss : 0.33721524477005005
# Epochs : 29 [57600/60000 (96.0) %, Loss : 0.40899449586868286
# Test Loss : 0.0085, Accuracy : 91.68
# Epochs : 30 [0/60000 (0.0) %, Loss : 0.42453381419181824
# Epochs : 30 [6400/60000 (10.666666666666666) %, Loss : 0.4461800158023834
# Epochs : 30 [12800/60000 (21.333333333333332) %, Loss : 0.16386564075946808
# Epochs : 30 [19200/60000 (32.0) %, Loss : 0.30604273080825806
# Epochs : 30 [25600/60000 (42.666666666666664) %, Loss : 0.2765991687774658
# Epochs : 30 [32000/60000 (53.333333333333336) %, Loss : 0.4456961154937744
# Epochs : 30 [38400/60000 (64.0) %, Loss : 0.3088054060935974
# Epochs : 30 [44800/60000 (74.66666666666667) %, Loss : 0.2960761785507202
# Epochs : 30 [51200/60000 (85.33333333333333) %, Loss : 0.49721765518188477
# Epochs : 30 [57600/60000 (96.0) %, Loss : 0.30753570795059204
# Test Loss : 0.0084, Accuracy : 91.8