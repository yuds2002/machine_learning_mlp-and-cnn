import torchvision
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn  # Layers
import torch.nn.functional as F # Activation Functions
import torch.optim as optim

def read_file(file):
    import pickle
    with open(file,'rb') as f:
        dict = pickle.load(f, encoding='latin1')
    return dict

dataset = torchvision.datasets.CIFAR10

transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to Tensor
    # Normalize Image to [-1, 1] first number is mean, second is std deviation
    transforms.Normalize((0.5,), (0.5,)) 
])

trainset = dataset(root='./data', train=True,
                                      download=True, transform=transform)

testset = dataset(root='./data', train=False,
                                      download=True, transform=transform)

BATCH_SIZE = 64
train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,shuffle=True)

test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,shuffle=False)

classes = trainset.classes
#print("num classes:",len(classes))
img, label = trainset[0]
#img_np = img.detach().cpu().numpy()
#plt.imshow(img.permute(1,2,0), cmap='gray', interpolation='none')
#print('Label (numeric):', label)
#print('Label (textual):', classes[label])
#print("train:",len(trainset))
#print("test:",len(testset))

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        # Conv2d(input,output,filterSize)
        # filtersize = n = (n*n)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, len(classes))
        self.output = nn.LogSoftmax(dim=1)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.output(x)
        return x

# Identify device
device = ("cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Creat the model and send its parameters to the appropriate device
cnn = CNN().to(device)



# Test on a batch of data
with torch.no_grad():  # Don't accumlate gradients
    cnn.eval()  # We are in evalutation mode
    x = example_data.to(device)
    outputs = cnn(x)  # Alias for cnn.forward

    # Print example output.
    print(torch.exp(outputs[0]))


# Define the training and testing functions
def train(net, train_loader, criterion, optimizer, device):
    net.train()  # Set model to training mode.
    running_loss = 0.0  # To calculate loss across the batches
    for data in train_loader:
        inputs, labels = data  # Get input and labels for batch
        inputs, labels = inputs.to(device), labels.to(device)  # Send to device
        optimizer.zero_grad()  # Zero out the gradients of the ntwork i.e. reset
        outputs = net(inputs)  # Get predictions
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()  # Propagate loss backwards
        optimizer.step()  # Update weights
        running_loss += loss.item()  # Update loss
    return running_loss / len(train_loader)

def test(net, test_loader, device):
    net.eval()  # We are in evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Don't accumulate gradients
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device) # Send to device
            outputs = net(inputs)  # Get predictions
            _, predicted = torch.max(outputs.data, 1)  # Get max value
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # How many are correct?
    return correct / total

cnn = CNN().to(device)

LEARNING_RATE = 0.006
MOMENTUM = 0.9


# Define the loss function, optimizer, and learning rate scheduler
criterion = nn.NLLLoss()
optimizer = optim.SGD(cnn.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

NUM_EPOCHS = 15

print("learning rate ",LEARNING_RATE)
print("batch size ",BATCH_SIZE)
print("momentum",MOMENTUM)
#print("num epochs ",NUM_EPOCHS)

count = 1
test_acc = 0 
for i in range(NUM_EPOCHS):
    train_loss = train(cnn, train_loader, criterion, optimizer, device)
    test_acc = test(cnn, test_loader, device)
    print(f"Epoch {i+1}: Train loss = {train_loss:.4f}, Test accuracy = {test_acc:.4f}")
    count+=1
    if test_acc >= 0.65:
        break
    


with torch.no_grad():  # Don't accumlate gradients
    cnn.eval()  # We are in evalutation mode
    plt.imshow(example_data[0][0], interpolation='none')
    x = example_data.to(device)
    outputs = cnn(x)  # Alias for cnn.forward

    # Print example output.
    print(torch.exp(outputs[0]))
    print(f'Prediction: {classes[torch.max(outputs, 1)[1][0]]}')