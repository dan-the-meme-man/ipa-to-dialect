"""train.py trains the model."""
import os
import json
import torch
from random import shuffle
from model import DialectNN
from sklearn.metrics import classification_report

if not os.path.exists('models'):
    os.mkdir('models')

epochs = 200 # probably make big - small dataset, cautious of overfitting
batch_size = 1 # probably leave as 1 - small dataset
learning_rate = 0.01 # just have to experiment
weight_decay = 0.001 # prevents overfitting
dim = 128 # just have to experiment
max_length = 275
criterion = torch.nn.CrossEntropyLoss()

# maps letters to integer indices
vocab = json.load(open(os.path.join('vocab', 'vocab_indices.json')))
vocab = {v: k for k, v in vocab.items()}
vocab_size = len(vocab) + 1
pad_id = len(vocab)

model = DialectNN(vocab_size=vocab_size, dim=dim, max_length=max_length)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay
)

#print(model(torch.rand(1, 100, 64)).shape)
#print(model)

train_data = json.load(open(os.path.join('splits', 'train.json')))
val_data = json.load(open(os.path.join('splits', 'val.json')))

# training loop
for e in range(epochs):
    
    print(f'Epoch {e+1}/{epochs}')
    
    shuffle(train_data) # randomize order
    
    num_batches = len(train_data) // batch_size
    
    model.train()
    
    for i in range(num_batches):
        
        # preprocess data
        train_items = train_data[i * batch_size : (i + 1) * batch_size]
        
        texts = [list(train_item[0].replace(' ', '')) for train_item in train_items]
        
        encoded_texts = [[int(vocab[c]) for c in text] for text in texts]
        
        for encoded_text in encoded_texts:
            while len(encoded_text) < max_length:
                encoded_text.append(pad_id)

        encoded_texts = torch.tensor(encoded_texts)
        
        labels = torch.tensor([[train_item[1] for train_item in train_items]])
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # feed into model        
        output = model(encoded_texts)
        
        # calculate loss
        loss = criterion(output, torch.tensor([labels]))
        
        # figure out why we did badly if we did
        loss.backward()
        
        # update weights
        optimizer.step()
        
        # log message
        print(f'Batch {i+1:3}/{num_batches}: loss {loss.item():.4f}')
        
    # dev set evaluation
    model.eval()
    preds = []
    labels = []
    for j in range(len(val_data)):
        
        # preprocess data
        train_item = val_data[j:j+1]
        
        texts = [list(train_item[0].replace(' ', '')) for train_item in train_items]
        
        encoded_texts = [[int(vocab[c]) for c in text] for text in texts]
        
        for encoded_text in encoded_texts:
            while len(encoded_text) < max_length:
                encoded_text.append(pad_id)

        encoded_texts = torch.tensor(encoded_texts)
        
        # true class
        label = [train_item[1] for train_item in train_items][0]
        labels.append(label)
        
        # predicted class  
        pred = torch.argmax(model(encoded_texts)[0]).item()
        preds.append(pred)
    
with open(os.path.join('models', f'model_{epochs}_{batch_size}_{learning_rate}_{weight_decay}_{dim}.txt'), 'w+') as f:
    f.write(classification_report(labels, preds))
        
torch.save(model, os.path.join('models', f'model_{epochs}_{batch_size}_{learning_rate}_{weight_decay}_{dim}.pt'))