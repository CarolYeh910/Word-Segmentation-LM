from preprocess import read_file, tag_to_ix, get_tag
from BiLSTM_CRF import *
import torch
import numpy as np
import matplotlib.pyplot as plt

content1, label1 = read_file('msr_test_gold.utf8')
content2, label2 = read_file('msr_training.utf8')


def train_data(content, label):
    train_data = []
    for i in range(len(label)):
        train_data.append((content[i], label[i]))
    return train_data


content = content1 + content2
label = label1 + label2
data = train_data(content, label)

word_to_ix = {}
for sentence, tags in data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)      # map words into numbers

net = torch.load('cws.model')
net.eval()
lines = []
text = open('msr_test.utf8', 'r', encoding='utf-8')
for line in text:
    line = line.strip('\n')
    line = line.strip(' ')
    lines.append(line)

tags = []
text = open('msr_test_gold.utf8', 'r', encoding='utf-8')
for line in text:
    line = line.strip('\n')
    line = line.strip(' ')
    letter_list = get_tag(line)
    tags.append(letter_list)

f1_score = 0.0
scores = []
f = open('res.txt', 'w', encoding='utf-8')
for j in range(len(tags)):
    seq = prepare_sequence(lines[j], word_to_ix)
    label = net(seq)[1]
    right, total, predict = 0, 0, 0
    sen = ''
    for i in range(len(label)):
        if i >= len(tags[j]):
            break
        sen += lines[j][i]
        if label[i] == 2 or label[i] == 3:
            predict = predict+1
            sen += '  '
        if tags[j][i] == 'E' or tags[j][i] == 'S':
            total = total+1
            if label[i] == tag_to_ix[tags[j][i]]:
                right = right+1
    sen += '\n'
    f.write(sen)
    score = 2*right/(total+predict)
    scores.append(score)
    f1_score += score
    if j % 100 == 0:
        print(f1_score)


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


print('F1 score of this model is %f' % (f1_score/len(tags)*100))
plt.figure(figsize=(20, 5))
plt.title('average reward: %f' % (f1_score/len(tags)*100))
plt.plot(moving_average(scores, 20))
plt.show()
