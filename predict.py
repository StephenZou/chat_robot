import argparse
import torch
import os

from BertForNLU import BertForNLU

parser = argparse.ArgumentParser()
parser.add_argument('--bert_path', help='config file', default='/Volumes/ExtraDisk/pre_model/bert-base-chinese')
parser.add_argument('--intent_file', help='intent file', default='/Volumes/ExtraDisk/code/github/chat_robot/data'
                                                                 '/intent.txt')
parser.add_argument('--domain_file', help='domain file', default='/Volumes/ExtraDisk/code/github/chat_robot/data'
                                                                 '/domain.txt')
parser.add_argument('--slot_file', help='slot file', default='/Volumes/ExtraDisk/code/github/chat_robot/data/slot.txt')
parser.add_argument('--gpu', help='which gpu to use', type=str, default='0')
