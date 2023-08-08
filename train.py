import argparse
import torch
import os

from BertForNLU import BertForNLU

parser = argparse.ArgumentParser()
parser.add_argument('--bert_path', help='config file', default='/Volumes/ExtraDisk/pre_model/bert-base-chinese')
parser.add_argument('--save_path', help='training file', default='/Volumes/ExtraDisk/code/github/chat_robot/train')
parser.add_argument('--train_text_file', help='training text file', default='/Volumes/ExtraDisk/code/github'
                                                                            '/chat_robot/data/train/text_train.txt')
parser.add_argument('--train_label_file', help='training label file', default='/Volumes/ExtraDisk/code/github'
                                                                              '/chat_robot/data/train/label_train.txt')
parser.add_argument('--valid_text_file', help='valid text file', default='/Volumes/ExtraDisk/code/github/chat_robot/data'
                                                                    '/val/text_val.txt')
parser.add_argument('--valid_label_file', help='valid label file', default='/Volumes/ExtraDisk/code/github/chat_robot'
                                                                           '/data/val/label_val.txt')
parser.add_argument('--intent_file', help='intent file', default='/Volumes/ExtraDisk/code/github/chat_robot/data'
                                                                 '/intent.txt')
parser.add_argument('--domain_file', help='domain file', default='/Volumes/ExtraDisk/code/github/chat_robot/data'
                                                                 '/domain.txt')
parser.add_argument('--slot_file', help='slot file', default='/Volumes/ExtraDisk/code/github/chat_robot/data/slot.txt')

parser.add_argument('--mode', help='run mode, train or predict', default='train')
parser.add_argument('--lr', type=float, default=8e-6)
parser.add_argument('--lr_warmup', type=float, default=200)
parser.add_argument('--bs', type=int, default=70)
parser.add_argument('--batch_split', type=int, default=3)
parser.add_argument('--eval_steps', type=int, default=2)
parser.add_argument('--n_epochs', type=int, default=30)
parser.add_argument('--max_length', type=int, default=90)
parser.add_argument('--n_jobs', type=int, default=1, help='num of workers to process data')

parser.add_argument('--gpu', help='which gpu to use', type=str, default='0')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu:0')

from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
from transformers import AdamW
import dataset
import utils
import traceback
from cls_trainer import ClsTrainer

train_path = os.path.join(args.save_path, 'train')
log_path = os.path.join(args.save_path, 'log')
logger = utils.get_logger(os.path.join(args.save_path, 'train.log'))


def save_func(epoch, device):
    filename = utils.get_ckpt_filename('model', epoch)
    torch.save(trainer.state_dict(), os.path.join(train_path, filename))


try:
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    for path in [train_path, log_path]:
        if not os.path.isdir(path):
            logger.info('cannot find {}, mkdiring'.format(path))
            os.makedirs(path)

    for i in vars(args):
        logger.info('{}: {}'.format(i, getattr(args, i)))
    tokz = BertTokenizer.from_pretrained(args.bert_path)

    _, intent2index, _ = utils.load_vocab(args.intent_file)
    _, domain2index, _ = utils.load_vocab(args.domain_file)
    _, slot2index, _ = utils.load_vocab(args.slot_file)
    train_dataset = dataset.EmotionDataset([[args.train_text_file, args.train_label_file]], tokz, intent2index,
                                           domain2index, slot2index, logger, max_lengths=args.max_length)
    valid_dataset = dataset.EmotionDataset([[args.valid_text_file, args.valid_label_file]], tokz, intent2index,
                                           domain2index, slot2index, logger, max_lengths=args.max_length)

    logger.info('Building models')
    bert_config = BertConfig.from_pretrained(args.bert_path)
    bert_config.num_intents = len(intent2index.keys())
    bert_config.num_domains = len(domain2index.keys())
    bert_config.num_slots = len(slot2index.keys())
    model = BertForNLU.from_pretrained(args.bert_path, config=bert_config)

    trainer = ClsTrainer(args, model, tokz, train_dataset, valid_dataset, log_path, logger, device)

    start_epoch = 0
    trainer.train(start_epoch, args.n_epochs, after_epoch_funcs=[save_func])
except:
    logger.error(traceback.format_exc())
