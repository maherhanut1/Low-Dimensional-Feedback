import math
import numpy as np
from tqdm import tqdm

from matplotlib import pyplot as plt

import torch



class Weights:
    
    def __init__(self, n, k, m, r, device='cpu'):
        
        self.W2 = torch.normal(0, 1/m, (m, k)).to(device)
        self.W1  = torch.normal(0, 1/k, (k, n)).to(device)
        
        self.P = torch.normal(0, 1/k, (r, m)).to(device)
        self.Q = torch.normal(0, 1/k, (k, r)).to(device)
    
    def clone_weights(self, other):
        
        self.W2 = other.W2.clone()
        self.W1 = other.W1.clone()
        self.P = other.P.clone()
        self.Q = other.Q.clone()
        
class TheorySimulator:
    
    def __init__(self, n, k, m, U, V_t, sig_mat_io, rank, lr, bw_lr, lamb, num_iterations, update_QP, device='cpu'):
        
        self.rank = rank
        self.update_QP = update_QP
        self.m = m
        
        #training params
        self.num_iterations = num_iterations
        self.lamb = lamb
        self.lr = lr
        self.bw_lr = bw_lr
        self.rank = rank
        self.lamb = lamb
        
        #InitWeight
        
        self.TheoryWeight = Weights(n, k, m, r=rank, device=device)
        self.SimulationWeight = Weights(n, k, m, r=rank, device=device)
        self.TheoryWeight.clone_weights(self.SimulationWeight)
        
        self.SimulationWeight.W2 = U @ self.TheoryWeight.W2.clone().to(device)
        self.SimulationWeight.W1 = self.TheoryWeight.W1.clone().to(device) @ V_t.to(device)

        # Truncated SVD with rank r to create bottleneck in Q and P
        A, B, C = torch.svd(self.SimulationWeight.W2)
        # Truncate to rank r
        A_r = A[:, :rank]  # m x r
        B_r = B[:rank]     # r
        C_r = C[:, :rank]  # k x r
        
        self.SimulationWeight.Q = C_r  # k x r
        self.SimulationWeight.P = (A_r * B_r.unsqueeze(0)).T  # r x m
        
        self.TheoryWeight.Q = self.SimulationWeight.Q.clone()
        self.TheoryWeight.P = self.SimulationWeight.P.clone()
        self.SimulationWeight.P = self.SimulationWeight.P @ U.T
        
        self.sig_mat_io = sig_mat_io
        self.U = U
        self.V_t = V_t
        
    
    def train(self, X, y):
        
        errors = []
        
        P = X.shape[-1]
        
        singulars_sim = []
        singulars_theory = []
        for i in tqdm(range(self.num_iterations)):
            
            # Full Simulation Update
            a1 = self.SimulationWeight.W1 @ X
            y_pred = self.SimulationWeight.W2 @ a1
            e = (y - y_pred) / P
            MSE = torch.mean((y - y_pred)**2)
            errors.append(MSE)
            W2_sim_grad = e @ a1.T
            W1_sim_grad = (self.SimulationWeight.Q @ self.SimulationWeight.P @ e) @ X.T
            # W1_sim_grad = (self.SimulationWeight.W2.T @ e) @ X.T
            self.SimulationWeight.W2 += self.lr * (W2_sim_grad) - self.lamb * self.SimulationWeight.W2
            self.SimulationWeight.W1 += self.lr * W1_sim_grad - self.lamb * self.SimulationWeight.W1
             
            if self.update_QP:
                E_sim = ((self.SimulationWeight.W2.T) - self.SimulationWeight.Q @ self.SimulationWeight.P)
                Q_sim_grad = E_sim @ self.SimulationWeight.P.T
                P_sim_grad = self.SimulationWeight.Q.T @ E_sim
                self.SimulationWeight.P += self.bw_lr * P_sim_grad #- self.lamb * self.SimulationWeight.P
                self.SimulationWeight.Q += self.bw_lr * Q_sim_grad #- self.lamb * self.SimulationWeight.Q
            
            
            # Theory Update:
            theory_w2w1 = self.TheoryWeight.W2 @ self.TheoryWeight.W1
            DS = (self.sig_mat_io - theory_w2w1)
            theory_W2_grad = DS @ self.TheoryWeight.W1.T
            W1_grad = self.TheoryWeight.Q @ self.TheoryWeight.P @ DS
            # W1_grad = self.TheoryWeight.W2.T @ DS
            
            self.TheoryWeight.W2 += self.lr * theory_W2_grad - self.lamb * self.TheoryWeight.W2
            self.TheoryWeight.W1 += self.lr * W1_grad - self.lamb * self.TheoryWeight.W1
            if self.update_QP:
                E_theory = ((self.TheoryWeight.W2.T) - self.TheoryWeight.Q @ self.TheoryWeight.P)
                Q_theory_grad = E_theory @ self.TheoryWeight.P.T
                P_theory_grad = self.TheoryWeight.Q.T @ E_theory
                self.TheoryWeight.P += self.bw_lr * P_theory_grad #- self.lamb * self.TheoryWeight.P
                self.TheoryWeight.Q += self.bw_lr *  Q_theory_grad #- self.lamb * self.TheoryWeight.Q
                self.bw_lr *= 0.999
            
            if i % 10 == 0:
                w2w1_sim = self.U.T @ self.SimulationWeight.W2@self.SimulationWeight.W1 @ self.V_t.T
                singulars_sim.append([(w2w1_sim[j,j] / self.sig_mat_io[j,j]).item() for j in range(self.m)])
                singulars_theory.append([(theory_w2w1[j,j] / self.sig_mat_io[j,j]).item() for j in range(self.m)])
            
        print(errors[-1])
        return singulars_sim, singulars_theory

if __name__ == "__main__":
    
    device = 'cuda'
    P = 100000
    
    X = torch.normal(0, 1, (128, P)).to(device)
    W_gt = torch.normal(0, 1, (64, 64)) @ torch.normal(0, 1, (64, 128)) / math.sqrt(64)
    W_gt = W_gt.to(device)
    y = W_gt @ X
    sigma_io = (1/(P)) * (y @ X.T)

    U, sig_io, V_t = torch.linalg.svd(sigma_io)
    print(sig_io)
    sig_mat_io = U.T @ sigma_io @ V_t.T
    
#    SimulationObj = TheorySimulator(128, 10, 128, U=U, V_t = V_t, sig_mat_io = sig_mat_io, rank=10, lr=5e-4, lamb=1e-8, num_iterations=20000, update_QP=True, device='mps')
    SimulationObj = TheorySimulator(128, 64, 64, U=U, V_t = V_t, sig_mat_io = sig_mat_io, rank=2, lr=2.5e-3, bw_lr=1e-3, lamb=1e-5, num_iterations=5000, update_QP=True, device='cuda')
    singulars_sim, singulars_theory = np.array(SimulationObj.train(X = X, y=y))



    fig, axs = plt.subplots(1, 1, figsize=(11.4, 5)) 
    s = 8
    lim = 100000
    # Create color maps
    blue_colors = plt.cm.Blues(np.linspace(0.7, 1,s))
    green_colors = plt.cm.Greens(np.linspace(0.65, 0.9,s))
    red_colors = plt.cm.Reds(np.linspace(0.4, 1,s))

    X = np.arange(len(singulars_sim[:, 0])) * 10

    for j in range(s):
        # ax.plot(singulars_BP[i, :, j], color=blue_colors[j], label='BP' if j == 0 else "")
        axs.plot(X[:lim], singulars_sim[:lim, j] , color=blue_colors[-j], linewidth=3, alpha=0.5)
        axs.plot(X[:lim], singulars_theory[:lim, j] , color=blue_colors[-j], linewidth=3, alpha=0.8, linestyle='--')
        # axs.plot(X[:lim], singulars_FA[:lim, j]  , color=green_colors[0], linewidth=2, linestyle='--')

    axs.set_xlabel('iterations', fontsize=23)
    axs.set_ylabel('$\hat{\sigma}$', fontsize=23)
    axs.legend(['Full Simulation', 'Theory'], loc='lower right', fontsize=12, ncol=1)

    plt.tight_layout()
    # plt.show()
    plt.savefig('normative_approach_r_less_than_m.svg')