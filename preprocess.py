import json
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split


def write_data(ds, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for d in ds:
            f.write(d+'\n')


if __name__ == '__main__':
    # 收集领域和意图信息
    domains = set()
    intents = set()
    slots = set()
    labels = []
    text = []
    # 进行BIO标记
    slots.add('O')
    token = BertTokenizer.from_pretrained('/Volumes/ExtraDisk/pre_model/bert-base-chinese')
    with open('data/train.json', 'r', encoding='utf-8') as f:
        train_ds = json.load(f)
        for data in train_ds:
            domains.add(data['domain'])
            intents.add(str(data['intent']))
            text.append(data['text'])
            # 使用BertTokenizer获取文本的token
            ids = token(data['text'])['input_ids'][1: -1]
            sent_label = [data['domain'], str(data['intent'])]+['O' for _ in range(len(ids))]
            sent_slots = data['slots']
            for slot in sent_slots.keys():
                slot_token = token(sent_slots[slot])['input_ids'][1: -1]
                slots.add('B-'+slot)
                slots.add('I-'+slot)
                slot_label = ['B-'+slot]+['I-'+slot for _ in range(len(slot_token)-1)]
                begin = ids.index(slot_token[0])
                sent_label[begin+2: begin+2+len(slot_token)] = slot_label
            labels.append(' '.join(sent_label))
    write_data(domains, 'data/domain.txt')
    write_data(intents, 'data/intent.txt')
    write_data(slots, 'data/slot.txt')
    text_train, text_val, label_train, label_val = train_test_split(text, labels, test_size=0.1, shuffle=True)
    write_data(text_train, 'data/train/text_train.txt')
    write_data(label_train, 'data/train/label_train.txt')
    write_data(text_val, 'data/val/text_val.txt')
    write_data(label_val, 'data/val/label_val.txt')

