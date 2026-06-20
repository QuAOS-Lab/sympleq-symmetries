# Attention Optimizer for Pauli Symmetries

This project searches for Pauli symmetries of Pauli-Sum Hamiltonians. The corresponding work is available at https://arxiv.org/abs/2605.30429.

## Example Usage

from Terminal:

CPU:
python Find_pauli_symmetry.py --iterid 200 --qubits 200 --model ising_chain --device cpu

GPU: 
python Find_pauli_symmetry.py --iterid 200 --qubits 200 --model ising_chain --device cuda



## Arguments

- `--iterid` - maximum number of optimization iterations.
- `--qubits` - number of qubits in the Hamiltonian.
- `--model` -  Hamiltonian model to generate. (Currently supports:
              - `ising_chain`
              - `ising_2d`
              - `toric`
              - `toric_magnetic` )
- `--device` - torch device, usually `cpu` or `cuda`



## Output

The script prints the model configuration, starts the timer, and then reports whether optimization succeeded before the iteration limit. On success it prints the rounded Pauli symmetry tensor:


## Setup

Dependencies: 

- `torch` (2.7.0+cu118)
- `transformers`
- `numpy`
- `scipy`
- `SympleQ` (https://github.com/QuAOS-Lab/SympleQ)



## Files

- `Find_pauli_symmetry.py` - runs the optimization loop and prints the discovered Pauli symmetry when successful.
- `config.py` - model and training configuration dataclass.
- `Set_Transformer.py` - Set Transformer model definition.
- `Loss_functions.py` - commutation, zero, binary, and regularization losses.
- `SympleQ_helpers.py` - converts SympleQ Hamiltonians into torch tableau input.
- `T_fixup.py` - T-Fixup initialization utilities.
- `requirements.txt` - Python dependencies.







