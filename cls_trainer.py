from optim import Adam, NoamOpt
import torch
import os
import torch.nn as nn
import torch.distributed
# import torch.tensor
from dataset import PadBatchSeq
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class ClsTrainer:
    def __init__(self, args, model, tokz, train_dataset, valid_dataset,
                 log_dir, logger, device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu:0'),
                 valid_writer=None):
        self.config = args
        self.device = device
        self.logger = logger
        self.log_dir = log_dir
        self.tokz = tokz
        self.train_writer = SummaryWriter(os.path.join(log_dir, 'train_cls'))
        if valid_writer is None:
            self.valid_writer = SummaryWriter(os.path.join(log_dir, 'valid_cls'))
        else:
            self.valid_writer = valid_writer
        self.model = model.to(device)
        self.criterion = nn.CrossEntropyLoss().to(device)

        base_optimizer = Adam(self.model.parameters(), lr=self.config.lr, weight_decay=0.01)
        self.optimizer = NoamOpt(self.model.config.hidden_size, 0.1, self.config.lr_warmup, base_optimizer)

        self.train_dataloader = DataLoader(
            train_dataset, shuffle=True, batch_size=self.config.bs, num_workers=self.config.n_jobs, pin_memory=True,
            collate_fn=PadBatchSeq(self.tokz.pad_token_id))

        self.valid_dataloader = DataLoader(
            valid_dataset, shuffle=False, batch_size=self.config.bs, num_workers=self.config.n_jobs, pin_memory=True,
            collate_fn=PadBatchSeq(self.tokz.pad_token_id))

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def _eval_train(self, epoch):
        self.model.train()

        loss, intent_acc, domain_acc, slot_acc, step_count = 0, 0, 0, 0, 0
        total = len(self.train_dataloader)
        TQDM = tqdm(enumerate(self.train_dataloader), desc='Train (epoch #{})'.format(epoch),
                    dynamic_ncols=True, total=total)

        for i, data in TQDM:
            text, intent, domain, slots = data['utt'].to(self.device), data['intent'].to(self.device), \
                                          data['domain'].to(self.device), data['slot'].to(self.device)
            mask = data['mask'].to(self.device)

            intent_logits, domain_logits, slot_logits = self.model(text, attention_mask=mask, return_dict=True)
            seq_len = slots.size(1)
            slots = slots.reshape(1, -1)
            slots = slots[slots != -1]
            batch_loss = (self.criterion(intent_logits, intent) + self.criterion(domain_logits, domain)
                          + self.criterion(slot_logits, slots)/seq_len)/3
            batch_intent_acc = (torch.argmax(intent_logits, dim=1) == intent).float().mean()
            batch_domain_acc = (torch.argmax(domain_logits, dim=1) == domain).float().mean()
            batch_slot_acc = (torch.argmax(slot_logits, dim=1) == slots).float().mean()

            full_loss = batch_loss / self.config.batch_split
            full_loss.backward()

            loss += batch_loss.item()
            intent_acc += batch_intent_acc.item()
            domain_acc += batch_domain_acc.item()
            slot_acc += batch_slot_acc.item()
            step_count += 1

            curr_step = self.optimizer.curr_step()
            lr = self.optimizer.param_groups[0]["lr"]
            # self.logger.info('epoch %d, batch %d' % (epoch, i))
            if (i + 1) % self.config.batch_split == 0:
                # update weights
                self.optimizer.step()
                self.optimizer.zero_grad()

                loss /= step_count
                intent_acc /= step_count
                domain_acc /= step_count
                slot_acc /= step_count

                self.train_writer.add_scalar('ind/loss', loss, curr_step)
                self.train_writer.add_scalar('ind/intent_acc', intent_acc, curr_step)
                self.train_writer.add_scalar('ind/domain_acc', domain_acc, curr_step)
                self.train_writer.add_scalar('ind/slot_acc', slot_acc, curr_step)
                self.train_writer.add_scalar('ind/lr', lr, curr_step)
                TQDM.set_postfix({'loss': loss, 'intent_acc': intent_acc, 'domain_acc': domain_acc, 'slot_acc': slot_acc})

                loss, intent_acc, domain_acc, slot_acc, step_count = 0, 0, 0, 0, 0

                # only valid on dev and sample on dev data at every eval_steps
                if curr_step % self.config.eval_steps == 0:
                    self._eval_test(epoch, curr_step)

    def _eval_test(self, epoch, step):
        self.model.eval()
        with torch.no_grad():
            all_intent_logits = []
            all_intent = []
            all_domain_logits = []
            all_domain = []
            all_slot_logits = []
            all_slot = []
            seq_len = 0
            for d_data in self.valid_dataloader:
                text, intent, domain, slots = d_data['utt'].to(self.device), d_data['intent'].to(self.device), \
                                              d_data['domain'].to(self.device), d_data['slot'].to(self.device)
                mask = d_data['mask'].to(self.device)
                intent_logits, domain_logits, slot_logits = self.model(text, attention_mask=mask, return_dict=True)
                all_intent.append(intent)
                all_intent_logits.append(intent_logits)
                all_domain.append(domain)
                all_domain_logits.append(domain_logits)
                seq_len += slots.size(1)
                slots = slots.reshape(1, -1)
                all_slot.append(slots)
                all_slot_logits.append(slot_logits)

            all_intent_logits = torch.cat(all_intent_logits, dim=0)
            all_intent = torch.cat(all_intent, dim=0)
            all_domain_logits = torch.cat(all_domain_logits, dim=0)
            all_domain = torch.cat(all_domain, dim=0)
            all_slot_logits = torch.cat(all_slot_logits, dim=0)
            all_slot = torch.cat(all_slot, dim=1).squeeze()
            all_slot = all_slot[all_slot != -1]
            loss = (self.criterion(all_intent_logits, all_intent).float() +
                    self.criterion(all_domain_logits, all_domain) +
                    self.criterion(all_slot_logits, all_slot)/seq_len)/3
            intent_acc = (torch.argmax(all_intent_logits, dim=1) == all_intent).float().mean()
            domain_acc = (torch.argmax(all_domain_logits, dim=1) == all_domain).float().mean()
            slot_acc = (torch.argmax(all_slot_logits, dim=1) == all_slot).float().mean()

            self.valid_writer.add_scalar('ind/loss', loss, step)
            self.valid_writer.add_scalar('ind/intent_acc', intent_acc, step)
            self.valid_writer.add_scalar('ind/domain_acc', domain_acc, step)
            self.valid_writer.add_scalar('ind/slot_acc', slot_acc, step)
            log_str = 'epoch {:>3}, step {}'.format(epoch, step)
            log_str += ', loss {:>4.4f}'.format(loss)
            log_str += ', intent_acc {:>4.4f}, domain_acc {:>4.4f}, slot_acc {:>4.4f}'\
                .format(intent_acc, domain_acc, slot_acc)
            self.logger.info(log_str)

        self.model.train()

    def train(self, start_epoch, epochs, after_epoch_funcs=[], after_step_funcs=[]):
        for epoch in range(start_epoch + 1, epochs):
            self.logger.info('Training on epoch'.format(epoch))
            self._eval_train(epoch)
            for func in after_epoch_funcs:
                func(epoch, self.device)
