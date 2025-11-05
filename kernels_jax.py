import numpy as np
from scipy import stats

import jax
import jax.numpy as jnp
from jax.nn import softmax


def k_wta(z, K):
    z_wta = jnp.zeros(len(z))
    top_k = jnp.argsort(z)[::-1][:K]
    z_wta = z_wta.at[top_k].set(z[top_k])
    return z_wta


def th_wta(z, th):
    """Thresholded WTA (allowing autograd)."""
    return jnp.where(z > th, z, 0)


def make_psi_rand_lin(J_PSI):
                  
    def psi(x):
        return J_PSI@x/np.sqrt(J_PSI.shape[0])
    
    def ker(x_1, x_2):
        return psi(x_1)@psi(x_2)
                  
    return psi, ker

def make_psi_rand_tanh(J_PSI):
                  
    def psi(x):
        return jnp.tanh(J_PSI@x)/np.sqrt(J_PSI.shape[0])
    
    def ker(x_1, x_2):
        return psi(x_1)@psi(x_2)
                  
    return psi, ker
    

def make_psi_rand_sgm(J_PSI):
                  
    def psi(x):
        return (jnp.tanh(J_PSI@x)+1)/(2*np.sqrt(J_PSI.shape[0]))
    
    def ker(x_1, x_2):
        return psi(x_1)@psi(x_2)
                  
    return psi, ker


def make_psi_rand_tanh_k_wta(J_PSI, K):
    
    def psi(x):
        return k_wta(jnp.tanh(J_PSI@x), K)/jnp.sqrt(K)
    
    def ker(x_1, x_2):
        return psi(x_1)@psi(x_2)
    
    return psi, ker


def make_psi_softmax(J_PSI, BETA):
    
    def psi(x):
        return softmax(BETA*jnp.tanh(J_PSI@x))
    
    def ker(x_1, x_2):
        return psi(x_1)@psi(x_2)
    
    return psi, ker


def make_psi_th_wta(J_PSI, TH):

    def psi(x, th):
        temp = th_wta(J_PSI@x, TH)
        return jnp.tanh(temp)
    
    def ker(x_1, x_2):
        return psi(x_1)@psi(x_2)
    
    return psi, ker


def make_ker_k_wta_approx(width):
    
    def ker(x_1, x_2):
        u_1 = x_1/jnp.linalg.norm(x_1)
        u_2 = x_2/jnp.linalg.norm(x_2)
        return jnp.exp((u_1@u_2 - 1)/width)
    
    return ker
