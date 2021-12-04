from preprocess import read_file, tag_to_ix
from BiLSTM_CRF import *
import torch
from torch import optim
import time
EMBEDDING_DIM = 5
HIDDEN_DIM = 4
epochs = 2

start = time.time()
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
            word_to_ix[word] = len(word_to_ix)     # map words into numbers


model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

for epoch in range(epochs):     # training
    for sentence, tags in data:
        model.zero_grad()

        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
        loss = model.neg_log_likelihood(sentence_in, targets)

        loss.backward()
        optimizer.step()
    print('epoch/epochs: {}/{}, loss:{:.6f}'.format(epoch+1, epochs, loss.data[0]))

end = time.time()
torch.save(model, 'cws.model')
torch.save(model.state_dict(), 'cws_all.model')
print((end-start)/60)
