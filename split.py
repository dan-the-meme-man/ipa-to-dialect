import os
import json
from sklearn.model_selection import train_test_split

texts = []
labels = []

regions = os.listdir('data')

for i in range(len(regions)):
    label = i # 0, 1, ...
    region = regions[i] # EasternNE, etc.
    files_in_region = os.listdir(os.path.join('data', region))
    for j in range(len(files_in_region)):
        file_name = os.path.join('data', region, files_in_region[j])
        loaded_json = json.load(open(file_name, 'r'))
        text = loaded_json['text']
        texts.append(text)
        labels.append(label)
        
X_train, temp_set, y_train, y_temp = train_test_split(texts, labels, test_size=0.3, random_state=42)

# Now, split the temporary set into validation and testing sets
X_val, X_test, y_val, y_test = train_test_split(temp_set, y_temp, test_size=0.5, random_state=42)

train_path = os.path.join('splits', 'train.json')
val_path = os.path.join('splits', 'val.json')
test_path = os.path.join('splits', 'test.json')

train_data = list(zip(X_train, y_train))
val_data = list(zip(X_val, y_val))
test_data = list(zip(X_test, y_test))

with open(train_path, 'w+') as f:
    json.dump(train_data, f)
    
with open(val_path, 'w+') as f:
    json.dump(val_data, f)
    
with open(test_path, 'w+') as f:
    json.dump(test_data, f)