import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
import psutil

from sklearn.model_selection import train_test_split

# Load the training dataset
train_data = np.loadtxt('train.csv', delimiter=',', dtype=np.float32, skiprows=1)
Xtrain = torch.from_numpy(train_data[:, 1:]) 
Ytrain = torch.from_numpy(train_data[:, [0]]).long().squeeze()

# Load the test dataset
test_data = np.loadtxt('test.csv', delimiter=',', dtype=np.float32, skiprows=1)
Xtest = torch.from_numpy(test_data[:, 1:])
Ytest = torch.from_numpy(test_data[:, [0]]).long().squeeze()

# Define the dataset class
class CustomDataset(Dataset):
    def __init__(self, data, label):
        self.len = data.shape[0]
        self.x_data = data
        self.y_data = label.long().squeeze()

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

# Mini-batch training
train_dataset = CustomDataset(Xtrain, Ytrain)
train_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True, num_workers=1)

# Build MLP model
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(784, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 64)
        self.fc4 = torch.nn.Linear(64, 25)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

model = Model()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

epoch_list, loss_list, acc_list, cpu_usage_list, memory_usage_list, flops_list = [], [], [], [], [], []

# Monitor CPU and memory usage
def get_resource_usage():
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_usage = psutil.virtual_memory().percent
    return cpu_usage, memory_usage

def count_flops(model, input_size):
    flops = 0
    x = torch.randn(1, *input_size)
    for layer in model.children():
        if isinstance(layer, torch.nn.Linear):
            flops += layer.in_features * layer.out_features
    return flops

def test():
    with torch.no_grad():
        y_pred = model(Xtest)
        loss = criterion(y_pred, Ytest)
        _, predicted = torch.max(y_pred, 1)
        accuracy = (predicted == Ytest).float().mean().item()
        print(f"Test Loss: {loss.item()}, Test Accuracy: {accuracy * 100:.2f}%")
        return loss.item(), accuracy

# Train the model
def train(epoch):
    start_time = time.time()
    train_loss = 0.0
    count = 0

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        y_pred = model(inputs)
        loss = criterion(y_pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        count = i

    cpu_usage, memory_usage = get_resource_usage()
    flops = count_flops(model, (784,))
    cpu_usage_list.append(cpu_usage)
    memory_usage_list.append(memory_usage)
    flops_list.append(flops)
    elapsed_time = time.time() - start_time

    print(f"Epoch {epoch}, Train Loss: {train_loss/count:.4f}, CPU: {cpu_usage}%, Memory: {memory_usage}%, FLOPs: {flops}, Time: {elapsed_time:.2f}s")

    test_loss, accuracy = test()
    epoch_list.append(epoch)
    loss_list.append(test_loss)
    acc_list.append(accuracy)

    # Draw loss and accuracy change curves
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epoch_list, loss_list, label='Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title('MLP Model Loss Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epoch_list, acc_list, label='Accuracy', color='green')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.title('MLP Model Accuracy Curve')
    plt.legend()

    plt.show()

if __name__ == '__main__':
    start_time = time.time()
    for epoch in range(10):
        train(epoch)
    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} seconds")
