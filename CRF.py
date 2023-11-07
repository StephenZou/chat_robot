import torch.nn as nn
import torch

START_TAG = "<start>"
STOP_TAG = "<stop>"


class CRF(nn.Module):
    def __init__(self, slot2idx):
        super(CRF, self).__init__()
        self.slot2idx = slot2idx
        self.slot_size = len(self.slot2idx)
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu:0')
        self.transitions = nn.Parameter(torch.randn(self.slot_size, self.slot_size))
        self.transitions.data[self.slot2idx[START_TAG], :] = -10000.
        self.transitions.data[:, self.slot2idx[STOP_TAG]] = -10000.

    @staticmethod
    def log_sum_exp(vec, batch_size):
        # max_score = vec[:, CRFLayers.argmax(vec)]
        max_score = torch.max(vec, dim=1).values
        max_score_broadcast = max_score.view(batch_size, -1).expand(batch_size, vec.size(1))
        return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast), dim=1))

    @staticmethod
    def argmax(vec):
        # 取一个列表的argmax
        _, idx = torch.max(vec, 1)
        return idx

    def _score_sentence(self, feats, tags, masks):
        # 给定输入序列和标记序列，求对应的分数
        batch_size = feats.size(0)
        score = torch.zeros(batch_size, 1).to(self.device)
        tags = torch.cat([torch.tensor([[self.slot2idx[START_TAG]] for _ in range(batch_size)], dtype=torch.long).to(self.device), tags], dim=1)
        # for x in masks:
        #     x[x.sum()-1] = 0
        #     x[0] = 0
        seq_len = feats.size(1)
        for i in range(1, seq_len-1):  # 对于序列中的每个位置
            feat = feats[:, i, :]
            # 累加当前位置的转移分数和发射分数，并将mask掉的部分分数置为0
            emit_score = feat[torch.arange(batch_size), tags[:, i-1]].view(-1, 1)
            emit_score = emit_score*masks[:, i].view(batch_size, -1)
            trans_score = self.transitions[tags[:, i], tags[:, i-1]].view(batch_size, 1)
            trans_score = trans_score*masks[:, i].view(batch_size, -1)
            score = score + trans_score + emit_score
        index = torch.sum(masks, dim=1)
        # 累加转移到STOP_TAG的转移分数
        score = score + self.transitions[self.slot2idx[STOP_TAG], index].view(batch_size, 1)
        return score

    def _forward_alg(self, logits, masks):
        batch_size = logits.size(0)
        init_alphas = torch.full((batch_size, self.slot_size), -10000.).to(self.device)
        init_alphas[:, self.slot2idx[START_TAG]] = 0.

        forward_var = init_alphas
        seq_len = logits.size(1)
        # for x in masks:
        #     x[x.sum()-1] = 0
        #     x[0] = 0
        for i in range(1, seq_len):
            logit = logits[:, i, :]
            # mask = masks[:, i] == 0
            alphas_t = []
            for next_tag in range(self.slot_size):
                emit_score = logit[:, next_tag].view(batch_size, -1).expand(batch_size, self.slot_size)
                # emit_score_copy = emit_score.clone()
                emit_score_copy = emit_score*masks[:, i].view(batch_size, -1).expand(batch_size, self.slot_size)
                # emit_score_copy.data[mask, :] = 0
                trans_score = self.transitions[next_tag].view(1, -1).expand(batch_size, -1)
                trans_score_copy = trans_score*masks[:, i].view(batch_size, -1).expand(batch_size, self.slot_size)
                # trans_score_copy = trans_score.clone()
                # trans_score_copy.data[mask, :] = 0
                next_tag_var = forward_var + trans_score_copy + emit_score_copy
                alphas_t.append(CRF.log_sum_exp(next_tag_var, batch_size).view(batch_size, -1))
            forward_var = torch.cat(alphas_t).view(batch_size, -1)
            # forward_var[mask, :] = 0
        terminal_var = forward_var + self.transitions[self.slot2idx[STOP_TAG]].view(1, -1).expand(batch_size, -1)
        alpha = CRF.log_sum_exp(terminal_var, batch_size)
        return alpha

    def _viterbi_decode(self, feats, masks):
        backpointers = []  # 初始化维特比算法的中间变量
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        # for x in masks:
        #     x[x.sum()-1] = 0
        #     x[0] = 0
        # 初始化维特比算法的中间变量
        init_vvars = torch.full((batch_size, self.slot_size), -10000.).to(self.device)
        init_vvars[:, self.slot2idx[START_TAG]] = 0
        # forward_var[i]中的元素对应 0：i-1 非规范化概率最大值
        forward_var = init_vvars
        for i in range(1, seq_len):  # 对于每个时间步 k
            feat = feats[:, i, :]
            # mask = masks[:, i] == 0
            feat = feat*masks[:, i].view(batch_size, -1).expand(batch_size, self.slot_size)
            bptrs_t = []  # 记录最大跳转位置
            viterbivars_t = []  # 记录下一个时间步 k+1 的维特比变量 u(k+1,V)

            for next_tag in range(self.slot_size):  # 对于每个标签 V
                # next_tag_var[i] 表示 u(k,y_i) + 转移分数
                # 这里不考虑发射分数，因为发射分数和前一个时间步的标签无关
                next_tag_var = forward_var + self.transitions[next_tag].view(1, -1).expand(batch_size, -1)

                # 取最大转移位置，并记录在 bptrs_t 中
                best_tag_id = CRF.argmax(next_tag_var)
                bptrs_t.append(best_tag_id)

                # 记录 u(k+1,y_i) - 发射分数
                viterbivars_t.append(next_tag_var[torch.arange(batch_size), best_tag_id].view(-1, 1))

            # 计算u(k+1,y_i)，也就是把发射分数加回来
            forward_var = (torch.cat(viterbivars_t, dim=1) + feat).view(batch_size, -1)
            bptrs_t = torch.stack(bptrs_t, dim=0).transpose(0, 1)
            backpointers.append(bptrs_t)

        # 计算跳转到 STOP_TAG 的非规范化序列
        # terminal_var = forward_var + self.transitions[self.slot2idx[STOP_TAG]].view(1, -1).expand(batch_size, -1)
        # best_tag_id = CRF.argmax(terminal_var)  # 得到概率最大的跳转位置，即最后一个位置的标签
        path_score = next_tag_var[torch.arange(batch_size), best_tag_id]

        # 从最后一个位置的标签获得整个标签序列
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[torch.arange(batch_size), best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert all(start == torch.full((batch_size,), self.slot2idx[START_TAG]).to(self.device))  # Sanity check
        best_path.reverse()
        best_path = torch.stack(best_path, dim=0).transpose(0, 1)
        return path_score, best_path

    def neg_log_likelihood(self, logits, slots, mask):
        forward_score = self._forward_alg(logits, mask)
        gold_score = self._score_sentence(logits, slots, mask)
        return torch.mean(forward_score - gold_score)

    def forward(self, logits, masks):
        score, best_seq = self._viterbi_decode(logits, masks)
        return score, best_seq
