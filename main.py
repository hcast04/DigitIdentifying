import torch 
import torch.nn as nn 
import torchvision.datasets as dsets 
import torchvision.transforms as transforms 
from torch.autograd import Variable 

# Hyper Parameters  
input_size = 28*28 # image size 
num_classes = 10 # number of digits
num_epochs = 5 # times to train the data set
batch_size = 100 # size of training batch
learning_rate = 0.001 

# MNIST Dataset (Images and Labels) 
train_dataset = dsets.MNIST(root ='./data', 
							train = True, 
							transform = transforms.ToTensor(), 
							download = True) 

test_dataset = dsets.MNIST(root ='./data', 
						train = False, 
						transform = transforms.ToTensor()) 

# Dataset Loader (Input Pipeline) 
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, 
										batch_size = batch_size, 
										shuffle = True) 

test_loader = torch.utils.data.DataLoader(dataset = test_dataset, 
										batch_size = batch_size, 
										shuffle = False) 





class LogisticRegression(nn.Module): 
	def __init__(self, input_size, num_classes): 
		super(LogisticRegression, self).__init__() 
		self.linear = nn.Linear(input_size, num_classes) 

	def forward(self, x): 
		out = self.linear(x) 
		return out 

model = LogisticRegression(input_size, num_classes) 

criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate) 

# Training the Model 
for epoch in range(num_epochs): 
	for i, (images, labels) in enumerate(train_loader): 
		images = Variable(images.view(-1, 28 * 28)) 
		labels = Variable(labels) 

		# Forward + Backward + Optimize 
		optimizer.zero_grad() 
		outputs = model(images) 
		loss = criterion(outputs, labels) 
		loss.backward() 
		optimizer.step() 

		if (i + 1) % 100 == 0: 
			print('Epoch: [% d/% d], Step: [% d/% d], Loss: %.4f'
				% (epoch + 1, num_epochs, i + 1, 
					len(train_dataset) // batch_size, loss.data)) 

# Test the Model 
correct = 0
total = 0
for images, labels in test_loader: 
	images = Variable(images.view(-1, 28 * 28)) 
	outputs = model(images) 
	_, predicted = torch.max(outputs.data, 1) 
	total += labels.size(0) 
	correct += (predicted == labels).sum() 

print('Accuracy of the model on the 10000 test images: % d %%' % ( 
			100 * correct / total)) 


