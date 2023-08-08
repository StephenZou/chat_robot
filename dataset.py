from torch.utils.data import Dataset
import torch


class EmotionDataset(Dataset):
    def __init__(self, paths, tokz, intent_vocab, domain_vocab, slot_vocab, logger, max_lengths=2048):
        self.logger = logger
        self.intent_vocab = intent_vocab
        self.domain_vocab = domain_vocab
        self.slot_vocab = slot_vocab
        self.max_lengths = max_lengths
        self.data = EmotionDataset.make_dataset(paths, tokz, intent_vocab, domain_vocab, slot_vocab, logger,
                                                max_lengths)

    @staticmethod
    def make_dataset(paths, tokz, intent_vocab, domain_vocab, slot_vocab, logger, max_lengths):
        logger.info('reading data from {}'.format(paths))
        dataset = []
        for path in paths:
            x_file = path[0]
            y_file = path[1]
            with open(x_file, 'r', encoding='utf8') as f:
                texts = [i.strip() for i in f.readlines() if len(i.strip()) != 0]
            with open(y_file, 'r', encoding='utf8') as f:
                labels = [i.strip() for i in f.readlines() if len(i.strip()) != 0]
            for t, l in zip(texts, labels):
                utt = tokz(t[: max_lengths])
                target = l.split()
                dataset.append([int(domain_vocab[target[0].lower()]),
                                int(intent_vocab[target[1].lower()]),
                                [int(slot_vocab[s.lower()]) for s in target[2:]],
                                utt['input_ids'],
                                utt['attention_mask']])
            # with open(path, 'r', encoding='utf8') as f:
            #     lines = [i.strip() for i in f.readlines() if len(i.strip()) != 0]
            #     lines = [i.split('\t') for i in lines]
            #     for label, utt in lines:
            #         # style, post, resp
            #         utt = tokz(utt[:max_lengths])
            #         dataset.append([int(label_vocab[label]),
            #                         utt['input_ids'],
            #                         utt['attention_mask']])
        logger.info('{} data record loaded'.format(len(dataset)))
        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        domain, intent, slot, utt, mask = self.data[idx]
        return {'domain': domain, 'intent': intent, 'slot': slot, "utt": utt, "mask": mask}


class PadBatchSeq:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        res = dict()
        res['domain'] = torch.LongTensor([i['domain'] for i in batch])
        res['intent'] = torch.LongTensor([i['intent'] for i in batch])
        max_len = max([len(i['utt']) for i in batch])
        res['slot'] = torch.LongTensor([i['slot'] + [-1] * (max_len - len(i['utt'])) for i in batch])
        res['utt'] = torch.LongTensor([i['utt'] + [self.pad_id] * (max_len - len(i['utt'])) for i in batch])
        res['mask'] = torch.LongTensor([i['mask'] + [self.pad_id] * (max_len - len(i['mask'])) for i in batch])
        return res


if __name__ == '__main__':
    from transformers import BertTokenizer

    bert_path = '/Volumes/ExtraDisk/pre_model/bert-base-chinese'
    text_file = '/Volumes/ExtraDisk/code/github/chat_robot/data/val/text_val.txt'
    label_file = '/Volumes/ExtraDisk/code/github/chat_robot/data/val/label_val.txt'
    intent_vocab = '/Volumes/ExtraDisk/code/github/chat_robot/data/intent.txt'
    domain_vocab = '/Volumes/ExtraDisk/code/github/chat_robot/data/domain.txt'
    slot_vocab = '/Volumes/ExtraDisk/code/github/chat_robot/data/slot.txt'
    with open(intent_vocab) as f:
        res = [i.strip() for i in f.readlines() if len(i.strip()) != 0]
    intent_vocab = dict(zip(res, range(len(res))))
    with open(domain_vocab) as f:
        res = [i.strip() for i in f.readlines() if len(i.strip()) != 0]
    domain_vocab = dict(zip(res, range(len(res))))
    with open(slot_vocab) as f:
        res = [i.strip() for i in f.readlines() if len(i.strip()) != 0]
    slot_vocab = dict(zip(res, range(len(res))))


    class Logger:
        def info(self, s):
            print(s)


    logger = Logger()
    tokz = BertTokenizer.from_pretrained(bert_path)
    dataset = EmotionDataset([[text_file, label_file]], tokz, intent_vocab, domain_vocab, slot_vocab, logger)
    pad = PadBatchSeq(tokz.pad_token_id)
    print(pad([dataset[i] for i in range(5)]))
