import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """ simple linear layer followed by a non-linearity """

    def __init__(self, params, emb=True):
        super().__init__()

        self.n_embd = params.n_embd
        self.dropout = params.dropout
        self.net = nn.Sequential(
            nn.Linear(self.n_embd, self.n_embd),
            nn.Linear(self.n_embd, self.n_embd),
        )

    def forward(self, x):
        return self.net(x)


class MultiAttentionBlock(nn.Module):
    """ Multi Attention block: communication followed by computation """

    def __init__(self, params,):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.sa = nn.MultiheadAttention(embed_dim=params.n_embd,
                                        num_heads=params.n_head_enc,
                                        dropout=params.dropout, bias=True,
                                        batch_first=True, device=params.device)
        self.ffwd = FeedForward(params)
        self.ln = params.layernorm

        if self.ln:
            self.ln1 = nn.LayerNorm(params.n_embd)
            self.ln2 = nn.LayerNorm(params.n_embd)

    def forward(self, query, key, val, key_padding_mask=None):
        # print(query)

        if self.ln:
            query = query + self.sa(query, key, val,
                                    key_padding_mask=key_padding_mask,
                                    need_weights=False, attn_mask=None)[0]
            query = query + self.ffwd(query)
            query = query + self.ffwd(query)

        return query


class SetAttentionBlock(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.mab = MultiAttentionBlock(params)
        self.emb = params.n_embd
        self.qubits = params.n_qubits
        self.nq = self.qubits
        self.n_embd = self.emb

    def forward(self, x, xj, key_padding_mask=None):
        x = self.mab(x, x, x, key_padding_mask=key_padding_mask)
        return x


class PMA(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        # self.n_class= params.n_vocab
        self.num_seeds = params.seed_size
        self.num_heads_dec = params.n_head_dec
        self.n_embd = params.n_embd
        self.device = params.device
        self.n_qubits = params.n_qubits

        self.seed_vectors = nn.Parameter(
            torch.full((self.num_seeds, self.n_embd), 1.0, device=params.device)
        )

        self.mha = nn.MultiheadAttention(self.n_embd, self.num_heads_dec,
                                         dropout=params.dropout, bias=True, 
                                         batch_first=True, device=params.device)
        #self.inv_proj = nn.Linear(self.n_embd, 2 * self.n_qubits)

    def forward(self, x, xj, key_padding_mask_seeds=None):
        batch_size = x.size(0)  # Repeat seed vectors for batch
        seeds = self.seed_vectors.unsqueeze(0).repeat(batch_size, 1, 1)  # Shape: (batch, num_seed(op), n_class)
        out, _ = self.mha(seeds, x, xj, need_weights=False)
        return out


class LearnableSigmoidWithThreshold(nn.Module):
    def __init__(self, num_features, init_temp=0.1, init_thresh=0.5, eps=1e-12, max_exp=50):
        super().__init__()
        self.register_buffer("temperature", torch.ones(num_features) * init_temp)
        self.register_buffer("threshold", torch.ones(num_features) * init_thresh)
        self.eps = eps
        self.max_exp = max_exp
        self.initialized = True

    def forward(self, x):
        if not self.initialized:
            with torch.no_grad():
                # Compute mean per feature dimension
                data_mean = x.mean(dim=(1, 2)) if x.dim() == 3 else x.mean(dim=0)
                self.threshold.copy_(data_mean)
            self.initialized = True

        temp = self.temperature.view(1, 1, -1)      # Shape: (1, 1, C)
        thresh = self.threshold.view(1, 1, -1)

        temp = torch.clamp(temp, min=self.eps)

        # Clamp exponent input to avoid overflow
        exp_input = -(x - thresh) / temp
        exp_input = torch.clamp(exp_input, min=-self.max_exp, max=self.max_exp)
        return 1 / (1 + torch.exp(exp_input))


class SinLayer(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class SetTransformerModel(nn.Module):

    def __init__(self, params):
        super().__init__()  # Passing the properties of nn.module
        self.params = params
        self.n_embd = params.n_embd
        self.emb = params.n_embd
        self.ln = params.layernorm
        self.n_qubits = params.n_qubits

        #self.embed = nn.Embedding(self.n_embd, self.n_embd)
        self.proj = nn.Linear(2 * self.n_qubits, self.n_embd)
        self.inv_proj = nn.Linear(self.n_embd, 2 * self.n_qubits)
        self.encoder_blocks = nn.Sequential(*[SetAttentionBlock(params) for _ in range(params.n_layer)])

        # Decoder block with pooling set-attention
        self.decoder_blocks = nn.Sequential(
            PMA(params),
            SetAttentionBlock(params)
        )

        if self.ln:
            self.ln_f = nn.LayerNorm(int(self.n_embd))

        self.sin = SinLayer()
        self.sigmoid_layer = LearnableSigmoidWithThreshold(int(2 * self.n_qubits), init_temp=0.1, init_thresh=0.5)

    def forward(self, idx1, step, prev_idx=None, list_pos=None, update_paulis=None):

        device = self.params.device
        pauli_update_step = self.params.pauli_update_step

        I = torch.eye(int(self.n_embd / 2), dtype=torch.int64, device=device)  # Identity n x n
        O = torch.zeros((int(self.n_embd / 2), int(self.n_embd / 2)), dtype=torch.int64, device=device)  # Zero n x n

        # Construct J as block matrix
        top = torch.cat((O, I), dim=1)  # [0, I]
        bottom = torch.cat((I, O), dim=1)  # [I, 0]
        J = torch.cat((top, bottom), dim=0)  # (B, T, 2n)
        J = J.to(device).float()      # (2n, 2n)

        B, T, Q = idx1.shape
        if step == 0:
            idx1 = idx1
        else:
            idx1 = prev_idx
            ns_p = list_pos

        if step % pauli_update_step == 0 and update_paulis == 1 and self.updated_this_step:
            _, T, Q = idx1.shape

            if T < 100:
                increment = 40
            elif T > 5000:
                increment = T / 100
            else:
                increment = T / 10

            for _ in range(int(increment)):
                perm = ns_p[torch.randperm(ns_p.size(0))]

                # idx1 = idx1[:, perm, :]
                if perm.shape[0] == 0:
                    idx1 = idx1

                elif perm.shape[0] == 1:
                    p1 = perm[:1]
                    p2 = torch.randint(0, idx1.shape[1], (1,))
                    new_elem = (idx1[:, p1, :] + idx1[:, p2, :]) % 2  # shape (1, C# print(new_elem)
                    # new_elem = new_elem.unsqueeze(1)
                    idx1 = torch.cat([idx1, new_elem], dim=1)
                    idx1_flat = idx1[0]  # shape (T+1, C)
                    idx1_unique = torch.unique(idx1_flat, dim=0)
                    idx1 = idx1_unique.unsqueeze(0)
                    _, T, Q = idx1.shape

                else:
                    p1, p2 = perm[:2]
                    new_elem = (idx1[:, p1, :] + idx1[:, p2, :]) % 2  # shape (1, C)
                    new_elem = new_elem.unsqueeze(1)        # shape (1, 1, C)

                    idx1 = torch.cat([idx1, new_elem], dim=1)
                    idx1_flat = idx1[0]  # shape (T+1, C)
                    idx1_unique = torch.unique(idx1_flat, dim=0)
                    idx1 = idx1_unique.unsqueeze(0)
                    _, T, Q = idx1.shape

            for _ in range(int(increment//5)):
                p1, p2 = torch.randint(0, idx1.shape[1], (2,))
                new_elem = (idx1[:, p1, :] + idx1[:, p2, :]) % 2  # shape (1, C# print(new_elem)
                new_elem = new_elem.unsqueeze(1)        # shape (1, 1, C)
                idx1 = torch.cat([idx1, new_elem], dim=1)
                idx1_flat = idx1[0]  # shape (T+1, C)
                idx1_unique = torch.unique(idx1_flat, dim=0)
                idx1 = idx1_unique.unsqueeze(0)
                _, T, Q = idx1.shape

                    

            # print(f'step:{step}, num_pauli: {T}')
            self.updated_this_step = True

        x = self.proj(idx1.float())

        assert Q == self.n_embd, f"Expected embedding dim {self.n_embd}, got {Q}"

        xj = torch.matmul(x, J)  # (B, T, 2n)  # (B, T, 2n)
        for block in self.encoder_blocks:
            x = block(x, x)

        for block in self.decoder_blocks:
            x = block(x, xj)  # (B, 2n_s, n_embd)

        x = self.ln_f(x)
        x = self.inv_proj(x)
        x = self.sin(x)
        x = self.sigmoid_layer(x)

        logits = x

        return logits, idx1
