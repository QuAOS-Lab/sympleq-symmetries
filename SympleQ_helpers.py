from sympleq.models.Ising import ising_chain_hamiltonian, ising_2d_hamiltonian
from sympleq.models.toric_code import ToricCode
import torch



def tableau_to_torch_input(model, qubits, periodic=True, device='cpu'):
    if model == 'ising_chain':
        n_spins = qubits
        H = ising_chain_hamiltonian(n_spins, J_zz=1, h_x=1, periodic=periodic)
        
    elif model == 'ising_2d':
        n_spins = int(qubits//2)
        H= ising_2d_hamiltonian(n_x=n_spins, n_y=2, J_zz=1, h_x=1, periodic=periodic)

    elif model == 'toric':
        Nx= 2
        Ny= int(qubits//(2*Nx))
        H = ToricCode(Nx, Ny, c_x=1, c_z=1, c_g=0, periodic=periodic).hamiltonian()

    elif model == 'toric_magnetic':
        Nx= 2
        Ny= int(qubits//(2*Nx))
        H = ToricCode(Nx, Ny, c_x=1, c_z=0, c_g=1, periodic=periodic).hamiltonian()

    else:
        raise ValueError(f"Model {model} not recognized. Choose from ising_chain, ising_2d, toric, or toric_magnetic.")

    x_tab, z_tab = H.x_exp, H.z_exp
    x_tab, z_tab = torch.tensor(x_tab), torch.tensor(z_tab)
    tab = torch.cat((x_tab, z_tab), dim=1)
    tab = tab.view(1, tab.shape[0], tab.shape[1]).int().to(device)
    
    
    return tab


if __name__ == "__main__":
    # Example usage
    model = 'ising_chain'
    qubits = 4
    device = 'cpu'
    result = tableau_to_torch_input(model, qubits, periodic=True, device=device)
    print(f'Built torch input for {model} with {qubits} qubits on {device}: {result.shape}', result)