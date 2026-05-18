import torch
import torch.nn as nn
from config import ModelParams


class Loss_fns(nn.Module):

    def __init__(self, n_qubits=2, params=ModelParams(), eps=1e-6):
        """
        Class wrapping all Loss Functions:
        - Commutation Loss
        - Zero Penalty
        - Binary Regularization
        - Linear Regularization
        """
        super().__init__()
        self.eps = eps
        self.n_qubits = n_qubits

    def finalize(self):
        self.n_qubits = self.n_qubits

    def comm_loss(self, logits, idx, params=ModelParams()):
        """
        Commutation loss

        """
        # print(logits.shape)
        device = logits.device
        qubits = self.n_qubits

        # Construct J as block matrix
        Id = torch.eye(qubits, dtype=torch.int64, device=device)
        Oz = torch.zeros((qubits, qubits), dtype=torch.int64, device=device)
        top = torch.cat((Oz, Id), dim=1)  # [0, I]
        bottom = torch.cat((Id, Oz), dim=1)  # [I, 0]
        J = torch.cat((top, bottom), dim=0).to(device).float()

        M = idx.float()
        MJ = torch.matmul(M, J)

        S_pred = logits
        comm_residual = torch.matmul(MJ, S_pred.transpose(1, 2))

        pi_const = torch.tensor(torch.pi, device=device) / 2
        comm_residual_mod2 = (torch.sin(pi_const * comm_residual)) ** 2
        loss = comm_residual_mod2.sum() + 1e-16

        flattened = torch.round(comm_residual_mod2[0]).view(-1)
        indices = torch.nonzero(flattened, as_tuple=False).flatten()

        return loss, indices

    def zero_penalty(self, logits):
        """
        Zero penalty term.
        logits: (B, T, C)
        pauli: scalar or tensor
        """

        logits = logits.mean(0)  # Mean across batch

        penalty = (1) / (logits.sum())**2

        return penalty

    def binary_reg(self, logits):
        """
        Binary regularization term.
        """

        logits = logits.mean(0)
        if self.n_qubits < 20:
            reg= torch.abs(logits * (1 - logits)).sum()
        else:
            reg = torch.abs(logits * (1 - logits)).sum()
        return reg

    def reg2(self, logits, idx, params=ModelParams(), eps=1e-8):
        """
        Linear regularization term.
        """
        device = logits.device
        qubits = self.n_qubits

        Id = torch.eye(qubits, dtype=torch.int64, device=device)
        Oz = torch.zeros((qubits, qubits), dtype=torch.int64, device=device)
        top = torch.cat((Oz, Id), dim=1)  # [0, I]
        bottom = torch.cat((Id, Oz), dim=1)  # [I, 0]
        J = torch.cat((top, bottom), dim=0).to(device).float()

        M = idx.float()
        MJ = torch.matmul(M, J)

        S_pred = logits

        comm_residual = torch.matmul(MJ, S_pred.transpose(1, 2))

        B, T, C = comm_residual.shape
        comm_mask = MJ.sum(2).view(B, T, C)

        comm_mask_transformed = comm_mask - comm_mask % 2
        comm_residual_masked = torch.abs(comm_residual - comm_mask_transformed)

        loss = comm_residual_masked.sum()

        return loss

    def total_loss(self, step, comm_loss, zero_penalty, binary_reg, reg2_loss, prev_gamma_reg, alpha_prev):
        """
        Combine losses with weights.
        """

        # Initialize weights for Commutation Loss and Linear Regularization based on their initial values

        if step == 0:
            r = reg2_loss / (comm_loss + 1e-12)
            # print(r, comm_loss, reg2_loss)
            if r > 10:
                alpha = 10
                prev_gamma_reg = 1

            else:
                alpha = 1
                prev_gamma_reg = 10

        else:
            alpha = alpha_prev
            prev_gamma_reg = prev_gamma_reg

        # Ramp Down for Linear Regularization Weights

        if step % 200 == 0:
            gamma_reg = prev_gamma_reg
            if step > 0:
                gamma_reg = prev_gamma_reg - prev_gamma_reg / 3
        else:
            gamma_reg = prev_gamma_reg

        # Fixed weights for Zero Penalty and Binary Regularization
        beta = 1
        beta_reg = 1

        loss = alpha * comm_loss + beta * zero_penalty + beta_reg * binary_reg + prev_gamma_reg * reg2_loss

        total = loss

        alpha_prev = alpha
        prev_gamma_reg = gamma_reg

        return total, prev_gamma_reg, alpha_prev
