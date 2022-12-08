import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.transforms as transforms
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import Convformer as CF
from tqdm import tqdm
import time
import os
import pandas as pd


#these parameters are determined by the shape of the data (except for kernel_size, thats just always 3)
image_size = (32,32)
channels = 3 
kernel_size = 3

#set batch size
batch_size = 512


def get_model(block_type, depth, patch_size = (4,4)):
    dim = patch_size[0] * patch_size[1] *3+2 #+2 for the position embedding
    return CF.Model(image_size = image_size, patch_size = patch_size, dim = dim, hidden_dim = dim, kernel_size = kernel_size, indim = dim, outdim = dim, numblocks = depth, block_type = block_type)

params = []
depths = [1,6,9]
block_types = ['concat', 'mm', 'inline', 'pct']
pss = [2,4]
for ps in pss:
    for block_type in block_types:
        for depth in depths:
            params.append({'block_type':block_type, 'depth':depth, 'patch_size':(ps,ps)})

params.append({'block_type':'concat', 'depth':20, 'patch_size':(4,4)})





'''
get on gpu
'''


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


'''
load data
'''


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


 

'''
define training
'''
def train(model, device, epochs, trainloader, testloader, verbose = False):

    start_time = time.time()
    model= nn.DataParallel(model)
    model = model.to(device)    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    lambda1 = lambda epoch: 0.89**(1.25*epoch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda1)
    train_accs = np.zeros(epochs)
    test_accs = np.zeros(epochs)
    for epoch in range(epochs):
        if epoch > 2 and train_acc < 0.15:
            print('Model not learning, ended training to save comp')
            return train_accs, test_accs, {'time':(time.time() - start_time)/epochs, 'params': sum(p.numel() for p in model.parameters())}
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]}')
        train_correct = 0
        train_total = 0    
        for batch_idx, (data, target) in enumerate(tqdm(trainloader)):
            if torch.cuda.is_available():
                data, target = data.to(device), target.to(device)
            model.train()
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs.data, 1)
            train_correct += (preds == target).sum().item()
            train_total += target.size(0)
            if batch_idx%100 == 0 and verbose:
                print(f'Loss: {loss.item()}')
        if epoch % 4 == 0:
            scheduler.step()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                # calculate outputs by running images through the network
                model.eval()
                outputs = model(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        train_acc, test_acc = train_correct/train_total, test_correct/test_total
        train_accs[epoch] = train_acc
        test_accs[epoch] = test_acc
        print(f'Epoch: {epoch + 1}, Train Acc: {train_acc}, Test Acc: {test_acc}')
    total_time = time.time() - start_time
    return train_accs, test_accs, {'time':total_time/epochs, 'params': sum(p.numel() for p in model.parameters())} 
        


'''
train models and save results
'''

newpath = './results'
if not os.path.exists(newpath):
    os.makedirs(newpath)

num_models_done = len(os.listdir('./results'))

epochs = 75
for param in params[num_models_done:]:
    print(param)
    model = get_model(param['block_type'],param['depth'],param['patch_size'])
    train_accs, test_accs, info = train(model = model, device = device, epochs = epochs, trainloader = trainloader, testloader = testloader)
    del model
    runtime, numparams = [None]*epochs, [None]*epochs
    runtime[0] = info['time']
    numparams[0] = info['params']
    df = pd.DataFrame({'train_accs':train_accs, 'test_accs':test_accs, 'runtime':runtime, 'numparams':numparams})
    path = f'/content/results/{param["block_type"]}_{param["depth"]}_{param["patch_size"]}.csv'
    df.to_csv(f'./results/{param["block_type"]}_{param["depth"]}_{param["patch_size"]}.csv', index = False)

print('good night!')



