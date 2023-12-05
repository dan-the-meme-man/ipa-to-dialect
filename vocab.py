import os
import json

s = set()

train_data = json.load(open(os.path.join('splits', 'train.json')))

with open ('vocab_no_space.txt', 'w+', encoding='utf-8') as f:
    for item in train_data:
        
        text = item[0]
        
        text = text.replace(' ', '')
        text = text.replace('\n', '')
        text = text.replace('\t', '')
        text = text.replace('\r', '')
        
        f.write(text + '\n')
        
        for char in text:
            s.add(char)
            
print(len(s))

s = list(s)
indices = list(range(len(s)))

vocab_indices = dict(zip(indices, s))

json.dump(vocab_indices, open('vocab_indices.json', 'w+'))