import torch
import torch.nn as nn

class DLNN(nn.Module):
    def __init__(self, input_size=511, liquid_size=23, output_size=2, leaky_rate_init=0.3, num_dendrites=4, dendrite_act='relu', # 'relu' 或 'sigmoid' 或 'tanh'
        gate_act='sigmoid', dropout_rate=0.1, use_layernorm=True,):
        super(DLNN, self).__init__()
        self.input_size   = input_size
        self.liquid_size  = liquid_size
        self.output_size  = output_size
        self.num_dendrites= num_dendrites

        self.leaky_rate = nn.Parameter(torch.tensor(leaky_rate_init))

        self.dend_act = getattr(torch, dendrite_act)
        self.gate_act = getattr(torch, gate_act)

        self.W_in      = nn.Parameter(torch.randn(liquid_size, num_dendrites, input_size))
        self.W_liquid  = nn.Parameter(torch.randn(liquid_size, num_dendrites, liquid_size))
        self.W_gate    = nn.Parameter(torch.randn(liquid_size, num_dendrites))

        self.b_liquid  = nn.Parameter(torch.zeros(liquid_size, 1))
        self.W_out     = nn.Parameter(torch.randn(output_size, liquid_size))
        self.b_out     = nn.Parameter(torch.zeros(output_size, 1))

        self.use_ln    = use_layernorm
        if use_layernorm:
            self.ln = nn.LayerNorm([liquid_size, 1])
        self.dropout   = nn.Dropout(dropout_rate)

        nn.init.xavier_uniform_(self.W_in,    gain=nn.init.calculate_gain(dendrite_act))
        nn.init.xavier_uniform_(self.W_liquid,gain=nn.init.calculate_gain(dendrite_act))
        nn.init.xavier_uniform_(self.W_out,   gain=1.0)

    def forward(self, x):
        # x: (batch, seq, input_dim)
        batch, seq_len, _ = x.size()
        liquid_state = torch.zeros(batch, self.liquid_size, 1, device=x.device)

        for t in range(seq_len):
            x_t = x[:, t, :]

            in_br = torch.einsum("ldi,bi->bld", self.W_in, x_t)
            in_br = self.dend_act(in_br)

            prev = liquid_state.squeeze(-1)
            rec_br = torch.einsum("ldh,bh->bld", self.W_liquid, prev)
            rec_br = self.dend_act(rec_br)

            gate = self.gate_act(self.W_gate).unsqueeze(0)

            merged_in  = (in_br  * gate).sum(-1, keepdim=True)
            merged_rec = (rec_br * gate).sum(-1, keepdim=True)

            liquid_in = merged_in + merged_rec + self.b_liquid.unsqueeze(0)
            if self.use_ln:
                liquid_in = self.ln(liquid_in)

            liquid_in = self.dropout(liquid_in)

            α = torch.sigmoid(self.leaky_rate)
            liquid_state = (1 - α) * liquid_state + α * torch.tanh(liquid_in)

        out = torch.matmul(self.W_out, liquid_state) + self.b_out
        return out.squeeze(-1)
