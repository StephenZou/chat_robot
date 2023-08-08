import os
import json
import random
import torch
import logging
import argparse
from torch.utils.checkpoint import checkpoint
from attrdict import AttrDict


def get_logger(filename, print2screen=True):
    logger = logging.getLogger(filename)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filename)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s][%(thread)d][%(filename)s][line: %(lineno)d][%(levelname)s] \
>> %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    if print2screen:
        logger.addHandler(ch)
    return logger


def load_vocab(vocab_file):
    with open(vocab_file) as f:
        res = [i.strip().lower() for i in f.readlines() if len(i.strip()) != 0]
    return res, dict(zip(res, range(len(res)))), dict(zip(range(len(res)), res))  # list, token2index, index2token


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def load_config(config_file):
    with open(config_file) as f:
        config = json.load(f)
        return AttrDict(config)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


def pad_sequence(sequences, batch_first=False, padding_value=0):
    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max([s.size(0) for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor


def checkpoint_sequential(functions, segments, *inputs):
    def run_function(start, end, functions):
        def forward(*inputs):
            for j in range(start, end + 1):
                inputs = functions[j](*inputs)
            return inputs
        return forward

    if isinstance(functions, torch.nn.Sequential):
        functions = list(functions.children())

    segment_size = len(functions) // segments
    # the last chunk has to be non-volatile
    end = -1
    for start in range(0, segment_size * (segments - 1), segment_size):
        end = start + segment_size - 1
        inputs = checkpoint(run_function(start, end, functions), *inputs)
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
    return run_function(end + 1, len(functions) - 1, functions)(*inputs)


def get_latest_ckpt(dir_name):
    files = [i for i in os.listdir(dir_name) if '.ckpt' in i]
    if len(files) == 0:
        return None
    else:
        res = ''
        num = -1
        for i in files:
            try:
                n = int(i.split('-')[-1].split('.')[0])
                if n > num:
                    num = n
                    res = i
            except ValueError:
                pass
        return res


def get_epoch_from_ckpt(ckpt):
    return int(ckpt.split('-')[-1].split('.')[0])


def get_ckpt_filename(name, epoch):
    return '{}-{}.ckpt'.format(name, epoch)


def get_ckpt_step_filename(name, step):
    return '{}-{}-step.ckpt'.format(name, step)


def f1_score(predictions, targets, average=True):
    def f1_score_items(pred_items, gold_items):
        common = Counter(gold_items) & Counter(pred_items)
        num_same = sum(common.values())

        if num_same == 0:
            return 0

        precision = num_same / len(pred_items)
        recall = num_same / len(gold_items)
        f1 = (2 * precision * recall) / (precision + recall)

        return f1
    
    scores = [f1_score_items(p, t) for p, t in zip(predictions, targets)]

    if average:
        return sum(scores) / len(scores)    

    return scores
