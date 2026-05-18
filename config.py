from dataclasses import dataclass
import torch


@dataclass
class ModelParams:
    # Model Parameters
    n_qubits: int = 0
    batch_size: int = 1
    n_embd: int = 2 * n_qubits  # Embedding dimension
    n_vocab: int = 2000
    n_head_enc: int = 1
    n_head_dec: int = 1
    chunk_size: int = 100
    n_layer: int = 2
    n_layers: int = n_layer
    dropout: float = 0.0
    bias: bool = True
    device: torch.device | str = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #device = torch.device("cpu")
    seed_size: int = 1

    # Training Parameters
    eval_interval: int = 1
    learning_rate: float = 1e-3
    loss_learning_rate: float = 1e-3
    max_iters: int = 401
    warm_iters: int = 351
    eval_iters: int = 1
    layernorm: bool = True  # Layer Norm

    pauli_update_step: int = 401

    def finalize(self):
        self.n_embd = self.n_embd
        self.device = torch.device(self.device)

