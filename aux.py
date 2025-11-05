import numpy as np


def get_flow(F, RX, e_0=None, e_1=None, u=None):  # make flow field for plotting
    if e_0 is None and e_1 is None:
        e_0 = np.array([1., 0])
        e_1 = np.array([0., 1])
        
    if u is None:
        u = 0*e_0
        
    flow = np.nan*np.zeros((len(RX), len(RX), 2))
    
    for cx_0 in range(len(RX)):
        for cx_1 in range(len(RX)):
            x = RX[cx_0]*e_0 + RX[cx_1]*e_1
            flow_orig = F(x, u)
            flow[cx_0, cx_1, 0] = flow_orig@e_0
            flow[cx_0, cx_1, 1] = flow_orig@e_1
            
    return flow


def get_flow_3(F, RX, e_0=None, e_1=None, e_2=None, u=None):  # make flow field for plotting
    if e_0 is None and e_1 is None:
        e_0 = np.array([1., 0, 0])
        e_1 = np.array([0., 1, 0])
        e_2 = np.array([0., 0, 1])
        
    if u is None:
        u = 0*e_0
        
    flow = np.nan*np.zeros((len(RX), len(RX), 3))
    
    for cx_0 in range(len(RX)):
        for cx_1 in range(len(RX)):
            x = RX[cx_0]*e_0 + RX[cx_1]*e_1
            flow_orig = F(x, u)
            flow[cx_0, cx_1, 0] = flow_orig@e_0
            flow[cx_0, cx_1, 1] = flow_orig@e_1
            flow[cx_0, cx_1, 2] = flow_orig@e_2
            
    return flow


def run_fwd_np(F, x_init, t, us=None):
    """Run dynamics forward."""
    
    dt = np.mean(np.diff(t))  # estimate dt
    D = len(x_init)
    
    xs = np.nan*np.zeros((len(t), len(x_init)))
    xs[0, :] = x_init.copy()  # initial condition
    
    for ct, t_ in enumerate(t[1:], 1):  # run dynamics
        
        if us is None:
            u = 0*x_init
        else:
            u = us[ct]
            
        dx = dt*F(xs[ct-1, :], u)
        
        xs[ct, :] = xs[ct-1, :] + dx
        
    return xs


def run_fwd_jnp(F, x_init, t, us=None):
    """Run dynamics forward."""
    
    dt = jnp.mean(jnp.diff(t))  # estimate dt
    D = len(x_init)
    
    xs = [x_init]
    
    for ct, t_ in enumerate(t[1:], 1):  # run dynamics
        
        if us is None:
            u = 0*x_init
        else:
            u = us[ct]
            
        dx = dt*F(xs[ct-1, :], u)
        
        xs.append(xs[ct-1, :] + dx)
        
    return jnp.array(xs)