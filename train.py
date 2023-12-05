"""train.py trains the model."""
import os
import json
import torch
from model import DialectNN

if not os.path.exists('models'):
    os.mkdir('models')

epochs = 200 # probably make big - small dataset, cautious of overfitting
batch_size = 1 # probably leave as 1 - small dataset
learning_rate = 0.001 # just have to experiment
weight_decay = 0.001 # prevents overfitting
dim = 32 # just have to experiment
max_length = 275
criterion = torch.nn.CrossEntropyLoss()

# maps letters to integer indices
vocab = json.load(open(os.path.join('vocab_indices.json')))
vocab = {v: k for k, v in vocab.items()}
vocab_size = len(vocab) + 1
pad_id = len(vocab)

model = DialectNN(vocab_size=vocab_size, dim=dim)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay
)

#print(model(torch.rand(1, 100, 64)).shape)
print(model)

train_data = json.load(open(os.path.join('splits', 'train.json')))
val_data = json.load(open(os.path.join('splits', 'val.json')))

for e in range(epochs):
    print(f'Epoch {e+1}/{epochs}')
    
    num_batches = len(train_data) // batch_size
    
    for i in range(len(train_data)):
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # preprocess data
        train_item = train_data[i]
        text = list(train_item[0].replace(' ', ''))
        encoded_text = [int(vocab[c]) for c in text]
        while len(encoded_text) < max_length:
            encoded_text.append(pad_id)
        encoded_text = torch.tensor(encoded_text)
        label = train_item[1]
        
        # feed into model
        output = model(encoded_text)
        
        # calculate loss
        loss = criterion(output, torch.tensor([label]))
        
        # figure out why we did badly if we did
        loss.backward()
        
        # update weights
        optimizer.step()
        
torch.save(model, os.path.join('models', f'model_{epochs}_{batch_size}_{learning_rate}_{weight_decay}_{dim}.pt'))