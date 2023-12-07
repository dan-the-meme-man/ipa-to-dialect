import os
import json
from collections import Counter

accents = os.listdir('data')
counters = dict.fromkeys(accents)

for accent in accents:
    counters[accent] = Counter()
    accent_files = os.listdir(os.path.join('data', accent))
    for file_name in accent_files:
        file_path = os.path.join('data', accent, file_name)
        text = json.load(open(file_path, 'r'))['text'].replace(' ', '').replace('\n', '').replace('\t', '').replace('\r', '')
        for char in text:
            counters[accent][char] = counters[accent].get(char, 0) + 1

with open('counts.json', 'w+') as f:
    json.dump(counters, f, indent=4)
    
def compare(accent1, accent2):
    for phone in counters[accent1]:
        try:
            if counters[accent1][phone] != counters[accent2][phone]:
                print(phone, ':', accent1, counters[accent1][phone], 'vs.', accent2, counters[accent2][phone])
        except:
            print(phone, accent1, counters[accent1][phone], 'vs.', accent2, 0)

for accent1 in counters:
    for accent2 in counters:
        if accent1 != accent2:
            print(accent1, accent2)
            compare(accent1, accent2)
            print()