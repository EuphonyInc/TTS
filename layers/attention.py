import torch
from torch import nn
from torch.nn import functional as F
from utils.generic_utils import sequence_mask


class BahdanauAttention(nn.Module):
    def __init__(self, annot_dim, query_dim, hidden_dim):
        super(BahdanauAttention, self).__init__()
        self.query_layer = nn.Linear(query_dim, hidden_dim, bias=True)
        self.annot_layer = nn.Linear(annot_dim, hidden_dim, bias=True)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, annots, query):
        """
        Shapes:
            - query: (batch, 1, dim) or (batch, dim)
            - annots: (batch, max_time, dim)
        """
        if query.dim() == 2:
            # insert time-axis for broadcasting
            query = query.unsqueeze(1)
        # (batch, 1, dim)
        processed_query = self.query_layer(query)
        processed_annots = self.annot_layer(annots)
        # (batch, max_time, 1)
        alignment = self.v(nn.functional.tanh(
            processed_query + processed_annots))
        # (batch, max_time)
        return alignment.squeeze(-1)


class AttentionRNNCell(nn.Module):
    def __init__(self, out_dim, annot_dim, memory_dim):
        super(AttentionRNNCell, self).__init__()
        self.rnn_cell = nn.GRUCell(out_dim + memory_dim, out_dim)
        self.alignment_model = BahdanauAttention(annot_dim, out_dim, out_dim)

    def forward(self, memory, context, rnn_state, annots,
                annot_lens=None):
        """
        Shapes:
            - memory: (batch, 1, dim) or (batch, dim)
            - context: (batch, dim)
            - rnn_state: (batch, out_dim)
            - annots: (batch, max_time, annot_dim)
            - annot_lens: (batch,)
        """
        # Concat input query and previous context context
        rnn_input = torch.cat((memory, context), -1)
        # Feed it to RNN
        # s_i = f(y_{i-1}, c_{i}, s_{i-1})
        rnn_output = self.rnn_cell(rnn_input, rnn_state)
        # Alignment
        # (batch, max_time)
        # e_{ij} = a(s_{i-1}, h_j)
        alignment = self.alignment_model(annots, rnn_output)
        if annot_lens is not None:
            mask = sequence_mask(annot_lens)
            mask = mask.view(memory.size(0), -1)
            alignment.masked_fill_(1 - mask, -float("inf"))
        # Normalize context weight
        alignment = F.softmax(alignment, dim=-1)
        # Attention context vector
        # (batch, 1, dim)
        # c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j
        context = torch.bmm(alignment.unsqueeze(1), annots)
        context = context.squeeze(1)
        return rnn_output, context, alignment
