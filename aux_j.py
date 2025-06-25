import jax.numpy as jnp


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
            
        dx = dt*F(xs[ct-1], u)
        
        xs.append(xs[ct-1] + dx)
        
    return jnp.array(xs)