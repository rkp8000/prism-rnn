import numpy as np
from scipy import stats
from scipy.special import softmax


def k_wta(z, K):
    z_wta = np.zeros(len(z))
    top_k = np.argsort(z)[::-1][:K]
    z_wta[top_k] = z[top_k]
    return z_wta


def th_wta(z, th):
    """Thresholded WTA (allowing autograd)."""
    return np.where(z > th, z, 0)


def make_psi_rand_lin(J_PSI):
    
    def psi(x):
        return (J_PSI@x)/np.sqrt(J_PSI.shape[0])
    
    def ker(x_1, x_2):
        return psi(x_1)@psi(x_2)
    
    return psi, ker


def make_psi_rand_tanh(J_PSI):
                  
    def psi(x):
        return np.tanh(J_PSI@x)/np.sqrt(J_PSI.shape[0])
    
    def ker(x_1, x_2):
        return psi(x_1)@psi(x_2)
                  
    return psi, ker
            
    
def make_psi_rand_sgm(J_PSI):
                  
    def psi(x):
        return (np.tanh(J_PSI@x)+1)/2*(np.sqrt(2/J_PSI.shape[0]))
    
    def ker(x_1, x_2):
        return psi(x_1)@psi(x_2)
                  
    return psi, ker


def make_ker_rand_sgm_theory():

    def ker(x_1, x_2):
        theta = np.arccos(x_1@x_2/(np.linalg.norm(x_1)*np.linalg.norm(x_2)))
        return 1-theta/np.pi

    return ker


def make_psi_rand_sgm_bias(J_PSI, BIAS):
                  
    def psi(x):
        return (np.tanh(J_PSI@x + BIAS)+1)/(2*np.sqrt(J_PSI.shape[0]))
    
    def ker(x_1, x_2):
        return psi(x_1)@psi(x_2)
                  
    return psi, ker


def make_psi_rand_k_wta(J_PSI, K):
    
    def psi(x):
        return k_wta(J_PSI@x/np.sqrt(J_PSI.shape[0]), K)/np.sqrt(K)
    
    def ker(x_1, x_2):
        return psi(x_1)@psi(x_2)
    
    return psi, ker


def make_psi_rand_tanh_k_wta(J_PSI, K):
    
    def psi(x):
        return k_wta(np.tanh(J_PSI@x), K)/np.sqrt(K)
    
    def ker(x_1, x_2):
        return psi(x_1)@psi(x_2)
    
    return psi, ker


def make_psi_rand_softmax(J_PSI, BETA):
    
    def psi(x):
        return softmax(BETA*J_PSI@x/np.sqrt(J_PSI.shape[0]))
    
    def ker(x_1, x_2):
        return psi(x_1)@psi(x_2)
    
    return psi, ker


def make_psi_rand_tanh_softmax(J_PSI, BETA):
    
    def psi(x):
        return softmax(BETA*np.tanh(J_PSI@x))
    
    def ker(x_1, x_2):
        return psi(x_1)@psi(x_2)
    
    return psi, ker


def make_psi_rand_th_wta(J_PSI, TH):

    def psi(x):
        temp = th_wta(J_PSI@x, TH)
        return np.tanh(temp)
    
    def ker(x_1, x_2):
        return psi(x_1)@psi(x_2)
    
    return psi, ker


def make_ker_k_wta_approx(width):
    
    def ker(x_1, x_2):
        u_1 = x_1/np.linalg.norm(x_1)
        u_2 = x_2/np.linalg.norm(x_2)
        return np.exp((u_1@u_2 - 1)/width)
    
    return ker
