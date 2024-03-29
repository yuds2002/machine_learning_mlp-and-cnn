import torchvision
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
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

BATCH_SIZE = 50
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

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten() # For flattening the 2D image
        
        self.fc1 = nn.Linear(3*32*32, 512) # input
        nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)  # First HL
        nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, len(classes))  
        self.dropout = nn.Dropout(0.22)
        self.output = nn.LogSoftmax(dim=1)
    def forward(self, x):
        x = self.flatten(x) # 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #Apply dropout
        x = self.dropout(x)
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
mlp = MLP().to(device)



# Test on a batch of data
with torch.no_grad():  # Don't accumlate gradients
    mlp.eval()  # We are in evalutation mode
    x = example_data.to(device)
    outputs = mlp(x)  # Alias for mlp.forward

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

mlp = MLP().to(device)
LEARNING_RATE = 0.0033
#LEARNING_RATE = 0.0055 gets 0.5659 with batch size 50,  1000 - 250 - 10
#LEARNING_RATE = 0.003 gets 0.5622 with batch size 50 at epoch 14,  1000 - 250 -250- 10
#LEARNING_RATE = 0.0033 gets 0.5772 with batch size 50,  512 - 256 - 10 lr_decay at 7 0.1
MOMENTUM = 0.95


# Define the loss function, optimizer, and learning rate scheduler
criterion = nn.NLLLoss()
optimizer = optim.SGD(mlp.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
lr_decay = optim.lr_scheduler.StepLR(optimizer, 6, 0.15)


NUM_EPOCHS = 15

print("learning rate ",LEARNING_RATE)
print("batch size ",BATCH_SIZE)
print("momentum",MOMENTUM)
#print("num epochs ",NUM_EPOCHS)

test_acc = 0 
for epoch in range(NUM_EPOCHS):
    train_loss = train(mlp, train_loader, criterion, optimizer, device)
    test_acc = test(mlp, test_loader, device)
    lr_decay.step()
    print(f"Epoch {epoch+1}: Train loss = {train_loss:.4f}, Test accuracy = {test_acc:.4f}")
    if test_acc >= 0.58:
        break
    



with torch.no_grad():  # Don't accumlate gradients
    mlp.eval()  # We are in evalutation mode
    plt.imshow(example_data[0][0])
    x = example_data.to(device)
    outputs = mlp(x)  # Alias for mlp.forward

    # Print example output.
    print(torch.exp(outputs[0]))
    print(f'Prediction: {classes[torch.max(outputs, 1)[1][0]]}')