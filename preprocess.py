import re
START_TAG = "<START>"
STOP_TAG = "<STOP>"
tag_to_ix = {"B": 0, "M": 1, "E": 2, "S": 3, START_TAG: 4, STOP_TAG: 5}


def get_word(sentence):     # map sentence into word list
    word_list = []
    sentence = ''.join(sentence.split(' '))
    for i in sentence:
        word_list.append(i)
    return word_list


def get_tag(sentence):      # map sentence into tag sequence
    tag_seq = []
    sentence = re.sub('  ', ' ', sentence)
    list = sentence.split(' ')
    for i in range(len(list)):
        if len(list[i]) == 1:
            tag_seq.append('S')
        elif len(list[i]) == 2:
            tag_seq.append('B')
            tag_seq.append('E')
        else:
            tag_seq.append('B')
            tag_seq.extend('M' * (len(list[i]) - 2))
            tag_seq.append('E')
    return tag_seq


def read_file(filename):
    content, label = [], []
    text = open(filename, 'r', encoding='utf-8')
    cnt = 0
    for line in text:
        cnt = cnt + 1
        if cnt > 20000:
            break
        line = line.strip('\n')
        line = line.strip(' ')
        content.append(get_word(line))
        label.append(get_tag(line))
    return content, label
