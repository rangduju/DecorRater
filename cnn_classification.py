#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models

#%%
torch.manual_seed(53113)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
batch_size = test_batch_size =32
kwargs = {'num_workers':0, 'pin_memory':True} if use_cuda else{}

#%%
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist_data', train=True, download=False,
    transform=transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))])),
    batch_size=batch_size, shuffle=True,**kwargs
)


test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist_data', train=False, download=False,
    transform=transforms.Compose(
        [transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,))])),
    batch_size=test_batch_size,
    shuffle=True, **kwargs
)

#%%
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    # forward和__init__要对齐，NotImplementedError是没有检测到forward
# %%
lr = 1e-3
momentum = 0.5
model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

#%%
def train(model, device, train_loader, optimizer, epoch, log_interval=100):
    model.train()
    for batch_idx,(data,target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)

        loss.backward()
        optimizer.step()
        if batch_idx % log_interval ==0:
            print("Train Epoch:{} [{}/{} ({:0f}%)]\tLoss:{:.6f}".format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                100.*batch_idx/len(train_loader),
                loss.item()
                ))

#%%
def test(model, devivce, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(devivce), target.to(devivce)
            output = model(data)
            test_loss +=F.nll_loss(output, target, reduction='sum').item()

            pred = output.argmax(dim=1, keepdim=True)
            correct +=pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

# %%
epochs = 2
for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)

save_model = True
if (save_model):
    torch.save(model.state_dict(),"mnist_cnn.pt") 
    #词典格式，model.state_dict()只保存模型参数

# %%
