
import argparse
import time

import torch
from transformers import get_cosine_schedule_with_warmup


from Set_Transformer import SetTransformerModel
from config import ModelParams
from Loss_functions import Loss_fns
from T_fixup import apply_tfixup
from SympleQ_helpers import tableau_to_torch_input


parser = argparse.ArgumentParser()
parser.add_argument('--iterid', type=int, required=True, help='max_iters')
parser.add_argument("--qubits", type=int, required=True, help="Number of qubits")
parser.add_argument("--model", type=str, required=True, help="Model type")
parser.add_argument("--device", type=str, required=True, help="Device type")


args = parser.parse_args()
params = ModelParams(n_qubits=args.qubits, n_embd=2 * args.qubits,
                     max_iters=args.iterid, warm_iters=int((args.iterid-1)/4),
                     device=args.device
                     )
params.finalize()
print(f"Configured model with {params.n_qubits} qubits")
print(f"Embedding size: {params.n_embd}")
print(f"Iterations: {params.max_iters}")
print(f"Warm Up: {params.warm_iters}")
print(f"Using Device: {params.device}")

loss_module = Loss_fns(n_qubits=args.qubits)
loss_module.finalize()



model = SetTransformerModel(params)

#Apply T-fixup Initiation
apply_tfixup(model.encoder_blocks, num_layers=params.n_layers, d_model=params.n_embd)
apply_tfixup(model.decoder_blocks, num_layers=1, d_model=params.n_embd)


model = model.to(params.device)
# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.99))


scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=params.warm_iters,
    num_training_steps=args.iterid
)


min_loss = []
imp_ratio = []
pauli_update_step = params.pauli_update_step

I = torch.eye(args.qubits, dtype=torch.int64, device= params.device)  # Identity n x n
O = torch.zeros((args.qubits, args.qubits), dtype=torch.int64,  device= params.device)  # Zero n x n
top = torch.cat((O, I), dim=1)  # [0, I]
bottom = torch.cat((I, O), dim=1)  # [I, 0]
J = torch.cat((top, bottom), dim=0).float()

inp_ham =  tableau_to_torch_input(args.model, args.qubits, periodic=True, device=params.device)

print(f'timer starts')

start_time = time.time()

for iter in range(args.iterid):
    xb = inp_ham

    if iter == 0:
        logits, prev_idx = model(xb, iter)

    else:
        logits, prev_idx = model(xb, iter, prev_idx=prev_idx, list_pos=indices, update_paulis=update_paulis)

    commloss, indices = loss_module.comm_loss(logits, prev_idx)
    penalty = loss_module.zero_penalty(logits)
    reg = loss_module.binary_reg(logits)
    reg2 = loss_module.reg2(logits, xb)
    _, pauli_update, _ = prev_idx.shape
    optimizer.zero_grad()

    if iter == 0:
        prev_gamma_reg = 10
        alpha_prev = 1
        pauli_update = xb.shape[1]
    loss, prev_gamma_reg, alpha_prev = loss_module.total_loss(step=iter, comm_loss=commloss, zero_penalty=penalty,
                                                              binary_reg=reg, reg2_loss=reg2,
                                                              prev_gamma_reg=prev_gamma_reg,
                                                              alpha_prev=alpha_prev)
    loss.backward()

    prev_loss = loss.detach().clone()  # convert to float for safe comparison
    if iter % pauli_update_step == 0:
        min_loss.append(prev_loss)
    else:
        if prev_loss < min_loss[-1]:
            min_loss.append(prev_loss)
        else:
            min_loss.append(min_loss[-1])

    if (iter + 1) % pauli_update_step == 0:
        model.updated_this_step = True
        #print(iter, model.updated_this_step)
        imp_ratio = torch.tensor([torch.abs(min_loss[i]- min_loss[i-1])
                                  for i in range(1, len(min_loss)-1)], device=params.device)
        ave_imp_ratio = torch.mean(imp_ratio)
        std_imp_ratio = torch.std(imp_ratio)
        ave_imp_ratio_10 = torch.mean(imp_ratio[90:])

        if ave_imp_ratio_10 < torch.abs(ave_imp_ratio - std_imp_ratio) +1e-4:
            update_paulis = 1
        else:
            update_paulis = 0
        min_loss = []

    else:
        update_paulis = 0
        model.updated_this_step = False

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    scheduler.step()

    # Stopping Criterion

    MJ = torch.matmul(xb.float(), J)

    S_pred = torch.round(logits)
    comm_residual = torch.matmul(MJ.float(), S_pred.transpose(1, 2))
    com_stop = (comm_residual % 2).sum()
    reg_stop = 0.25 * (args.qubits / 10)

    if com_stop == 0 and torch.any(torch.round(logits[0]) != 0) and reg < reg_stop:

        break


time_taken = time.time() - start_time

print()
if iter < args.iterid - 1:
    print(f"Time taken: {time_taken}")
    print()
    print(f"step {iter}: optimization successful")
    print()
    print(f'Found Pauli Symmetry:', S_pred)
    print()
else:
    print(f"Time taken: {time_taken}")
    print(f"step {iter}: optimization unsuccessful")
    print()



