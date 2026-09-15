import diffrax as dfx
import jax.numpy as jnp
import jax
import functools

#----------------------------------------------------------------------------------------
# Model definition
#----------------------------------------------------------------------------------------


def vf(t,y,args):
    """
    rhs of GLV expressed in log-abundances
    """
    r, A, m = args
    return y*(r - A @ y) + m


def vf_log(t,z,args):
    """
    rhs of GLV expressed in log-abundances
    """
    r, A, m = args
    x = jnp.exp(z)
    return r - A @ x + m / x

def jac(t, x, args):
    """
    Jacobian of GLV in linear abundances
    """
    r, A, m = args    
    J = - x[:, None] * A
    J = J.at[jnp.arange(x.size), jnp.arange(x.size)].add(r - A@x)
    return J


#----------------------------------------------------------------------------------------
# Integration schemes
#----------------------------------------------------------------------------------------

_SOLVE_DOC_TEMPLATE = """
Generates a trajectory of the GLV with parameters (r, A, m), sampled `N` times at
intervals of `dT`.

{description}

## Params
- `N` : number of recorded steps (integrator)
- `dT` : continuous time interval between recorded steps
- `r,A,m` : growth rates, interaction matrix, immigration rate
- `y0`: initial condition
- `rtol, atol` relative and absolute tolerances for the adaptive step size
- `threshold` : abundance that fall on or below this value will be frozen
- `max_steps` : maximum number of internal steps for the solver
"""

def make_adaptive_solve(vf, solver, trans=None, description=""):
    """
    Maker for different integration schemes
    
    ## Params
    - `vf` : vecor field, rhs of ODE
    - `solver` : a diffrax solver
    - `trans` : tuple of transform and anti-transform applied to state before and after integration
    """
    
    
    def _solve(N, dT, r, A, m, y0,rtol=1e-6,atol=1e-8,threshold=None,max_steps=None):
            
            if max_steps is None:
                max_steps = int(100*N*dT)

            if threshold is not None:
                threshold_ = trans[0](threshold) if trans is not None else threshold

                def vf_sticky(t, y, args):
                    raw = vf(t, y, args)
                    return jnp.where(y <= threshold_, 0., raw)

                term = dfx.ODETerm(vf_sticky)               
            else:
                term = dfx.ODETerm(vf)
    
            ts = jnp.arange(N)*dT
            y0_ = trans[0](y0) if trans is not None else y0
                
            saveat = dfx.SaveAt(ts=ts)
            pid = dfx.PIDController(rtol, atol)
    
            sol = dfx.diffeqsolve(term,solver,0,ts[-1],0.01,y0_,(r,A,m),saveat=saveat,stepsize_controller=pid,max_steps=max_steps)
    
            ts = sol.ts
    
            ys = trans[1](sol.ys) if trans is not None else sol.ys
    
            return ts, ys

    _solve.__doc__ = _SOLVE_DOC_TEMPLATE.format(description=description)

    return _solve


solve_log_RK45 = make_adaptive_solve(
    vf_log, dfx.Tsit5(), (jnp.log, jnp.exp),
    description=(
        "The integration uses an adaptive Runge-Kutta 4/5 in the space of "
        "log-abundances. Note: in the absence of migration, the `threshold`"
        "parameter should be set to absorb abundances at a small but finite value"
    ),
)
    
solve = solve_log_RK45
jit_solve = functools.partial(jax.jit, static_argnums=(0,9))(solve)     # for convenience

solve_RK45 = make_adaptive_solve(
    vf, dfx.Tsit5(), None, description=(
        "The integration uses an adaptive Runke-Kutta 4/5 in linear abundances."
        ),
    )

solve_implicit34 = make_adaptive_solve(
    vf, dfx.Kvaerno4(), None, description=(
        "The integration uses an adaptive and implicit Kvaerno 4/3 method in linear abundances. Abundances"
        "below `cutoff` (if given) are set to zero."
        ),
    )

#----------------------------------------------------------------------------------------
# Batching convenience  
#----------------------------------------------------------------------------------------

def jit_batch(solve_x, N, dT, rtol=1e-6, atol=1e-8, threshold=None, max_steps=None):
    """
    Wrap a glv solve function into a jitted, batched function of (r, A, m, y0).

    - r:  (B, N) or (B)     batched growth rates
    - A:  (B, N, N)         batched interaction matrices
    - m:  (B, N) or (B)     batched mortality/dilution rates
    - y0: (B, N)            batched initial conditions
    """
    def wrapped(r, A, m, y0):
        return solve_x(N, dT, r, A, m, y0, rtol=rtol, atol=atol, threshold=threshold, max_steps=max_steps)

    batched = jax.jit(jax.vmap(wrapped, in_axes=(0, 0, 0, 0)))
    return jax.jit(batched)


def jit_batch_solve(N, dT, rs, As, ms, y0s, rtol=1e-6, atol=1e-8, threshold=None, max_steps=None):
    return jit_batch(solve, N, dT, rtol, atol, threshold, max_steps)(rs, As, ms, y0s) 