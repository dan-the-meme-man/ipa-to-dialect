"""test.py trains the model."""
import os
import json
import torch
from sklearn.metrics import classification_report

max_length = 275

# maps letters to integer indices
vocab = json.load(open(os.path.join('vocab', 'vocab_indices.json')))
vocab = {v: k for k, v in vocab.items()}
vocab_size = len(vocab) + 1
pad_id = len(vocab)

test_data = json.load(open(os.path.join('splits', 'test.json')))

models = os.listdir('models')
for model_name in models:
    
    model = torch.load(os.path.join('models', model_name))

    # test set evaluation
    model.eval()
    preds = []
    labels = []
    for j in range(len(test_data)):
        
        # preprocess data
        test_items = test_data[j:j+1]
        
        texts = [list(test_item[0].replace(' ', '')) for test_item in test_items]
        
        encoded_texts = [[int(vocab[c]) for c in text] for text in texts]
        
        for encoded_text in encoded_texts:
            while len(encoded_text) < max_length:
                encoded_text.append(pad_id)

        encoded_texts = torch.tensor(encoded_texts)
        
        # true class
        label = [test_item[1] for test_item in test_items][0]
        labels.append(label)
        
        # predicted class  
        pred = torch.argmax(model(encoded_texts)[0]).item()
        preds.append(pred)
        
    with open(os.path.join('models', model_name[:-3] + '_test.txt'), 'w+') as f:
        f.write(classification_report(labels, preds))