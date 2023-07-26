import json
from nltk_utils import tokenize,stem, bag_of_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import sys
sys.path.insert(1, 'movies recommendation chatbot\models')
from chat_model import NeuralNet

with open("e:\data science\Projects\movies recommendation chatbot\chatbot data\data.json") as file:
    prompts = json.load(file)

all_words = []
tags = []
xy = []

for prompt in prompts['prompts']:
    tag = prompt['tag']
    tags.append(tag)
    for pattern in prompt["patterns"]:
        word = tokenize(pattern)
        all_words.extend(word)
        xy.append((word,tag))


ignored_words = ['?','!','.',',',"'s"]

all_words = [stem(word) for word in all_words if word not in ignored_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)


class ChatDataset(Dataset):
    def __init__(self):
        self.n_simples = len(X_train)
        self.X_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_simples
    
batch = 8
input_size = len(all_words)
hidden_size = 8
n_classes = len(tags)
dataset = ChatDataset()
train_loader = DataLoader(dataset= dataset,batch_size=batch, shuffle=True, num_workers=0)
n_epochs = 1000
learning_rate = 0.001

model = NeuralNet(input_size, hidden_size, n_classes)

loss_fun = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)

for epoch in range(n_epochs):
    for (words, labels) in train_loader:
        words = words.to(torch.float32)
        labels = labels.to(torch.long)
        outputs = model(words)
        loss = loss_fun(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'epoch{epoch+1}/{n_epochs}, loss={loss.item(): .4f}')


data = {
    "model": model.state_dict(),
    "input_size": input_size,
    "output_size": n_classes,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "movies recommendation chatbot\models\data.pth"
torch.save(data,FILE)
