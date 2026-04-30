import jax 
from jax import grad, jit, vmap, pmap, value_and_grad, jacfwd, jacrev, hessian
import matplotlib
import matplotlib.pyplot as plt
import os
import tqdm
import diffrax
import optax
import optimistix
import equinox

# Setting GPU
dtype = jax.numpy.float32
device = jax.devices("cuda")[0]

@jax.jit
def laplacian(V,dx,dy):
    lap = jax.numpy.zeros_like(V)
    lap = lap.at[1:-1, 1:-1].set((V[2:, 1:-1] - 2 * V[1:-1, 1:-1] + V[:-2, 1:-1]) / dx**2.0 + \
      (V[1:-1, 2:] - 2 * V[1:-1, 1:-1] + V[1:-1,:-2]) / dy**2.0)
    lap = lap.at[0, 1:-1].set(2*(V[1, 1:-1]- V[0, 1:-1]) / dx**2 + (V[0, 2:] - 2 * V[0, 1:-1] + V[0,:-2]) / dy**2.0)
    lap = lap.at[-1, 1:-1].set(2*(-V[-1, 1:-1]+ V[-2, 1:-1]) / dx**2 + (V[-1, 2:] - 2 * V[-1, 1:-1] + V[-1,:-2]) / dy**2.0)
    lap = lap.at[1:-1, 0].set((V[2:, 0] - 2 * V[1:-1, 0] + V[:-2, 0]) / dx**2.0 + 2*(V[1:-1, 1]- V[1:-1, 0]) / dy**2)
    lap = lap.at[1:-1, -1].set((V[2:, -1] - 2 * V[1:-1, -1] + V[:-2, -1]) / dx**2.0+ 2*(-V[1:-1,-1]+ V[ 1:-1,-2]) / dy**2)
    lap = lap.at[0,0].set(2 * (V[1, 0] - V[0, 0]) / dx**2.0 + 2 * (V[0, 1] - V[0, 0]) / dy **2)
    lap = lap.at[-1,-1].set(2 * (V[-2, -1] - V[-1, -1]) / dx**2.0 + 2 * (V[-1, -2] - V[-1, -1]) / dy **2)
    lap = lap.at[0,-1].set(2 * (V[1, -1] - V[0, -1]) / dx**2.0 + 2 * (V[0, -2] - V[0, -1]) / dy **2)
    lap = lap.at[-1,0].set(2 * (V[-2, 0] - V[-1, 0]) / dx**2.0 + 2 * (V[-1, 1] - V[-1, 0]) / dy **2)
    return lap

@jax.jit
def rhs(t, y, args, params):
    """2-D reaction-diffusion model of skin
    
    dT(x,y,t)/dt = D_T(x,y) * laplacian(T) + f(T, U)
    dU(x,y,t)/dt = g(T, U)
    
    for T = temperature and U = ulcer value
        x,y = positions
        t = time
        q_src(x,y,t) = reaction term for temperature
    
    Args:
        t: time
        y: state vector [N, M, 2]
            y = temperature
    
    Returns:
        rhs: right-hand side of the ODE
    """
    T = y

    dx = args['dx']
    dy = args['dy']
    dt = args['dt']
    D_T = args['D_T']
    h = args['h']
    T_amb = args['T_amb']

    q_src = params['q_src']

    laplacian_T = laplacian(T, dx,dy)
    
    dydt = D_T * laplacian_T + q_src - h * (T - T_amb)
    
    return dydt


def forward_solve(dt, y0, args,params):

    N = args['N']
    M = args['M']
    num_save_points = args.get('num_save_points', 10)

    t0 = 0.0
    tfinal = args['tfinal']

    print("\nBeginning timesteps")
    # for step in tqdm.tqdm(range(nsteps), desc="Time stepping"):      
    #   t = t0 + step * dt                                  
    #   dy_dt = rhs(t, y, dydt, args, params)
    #   y = y + dt * dy_dt

    def rhs_fn_wrapper(t, y, args):
        return rhs(t, y, args, params)

    term = diffrax.ODETerm(rhs_fn_wrapper)
    # Diffusion is stiff: for explicit Heun the stability limit is
    # dt <= dx^2 / (4 D_T) ~ 2.5e-6 here, so we use an L-stable implicit
    rtol, atol = 1e-4, 1e-4
    stepsize_controller = diffrax.PIDController(
        pcoeff=0.3, icoeff=0.4, rtol=rtol, atol=atol, dtmax=60.0
    )
    solver = diffrax.Heun()
    save_ts = jax.numpy.linspace(t0, tfinal, num_save_points)
    saveat = diffrax.SaveAt(ts=save_ts)
    #progress_meter = diffrax.TqdmProgressMeter()
    sol = diffrax.diffeqsolve(
        term,
        solver=solver,
        t0=t0,
        t1=tfinal,
        dt0=dt,
        y0=y0,
        args=args,
        max_steps=None,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        #progress_meter=progress_meter
    )

    ys = jax.numpy.asarray(sol.ys)

    fig, ax = plt.subplots(1, 1, dpi=250)
    cbar = None
    for i in range(len(sol.ts)):
        # Remove old colorbar before ax.clear(); clear() drops the axes the
        # colorbar attached to and leaves Colorbar.remove() in a bad state.
        if cbar is not None:
            cbar.remove()
            cbar = None
        ax.clear()

        # Compute per-frame color limits
        vmin = float(ys[i].min())
        vmax = float(ys[i].max())

        im = ax.imshow(ys[i], vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(im, ax=ax)

        if i == 0:
            ax.set_title("Initial Condition")
        else:
            ax.set_title(f"t = {sol.ts[i]:.3f}")

        plt.savefig(f"ulcer_{i}.png")

    return sol


def main():

    # Output directori
    #os.mkdir("data", exist_ok=True)
    print_arguments = True

    # **************** GROUND TRUTH ****************

    rho = 1050 # density
    C = 3770 # heat capacity
    rho_c = rho * C # rho * C
    k = 0.4 + 0.2 # thermal conductivity
    alpha = k / rho_c # Thermal diffusivity
    d_skin = 0.002 # thickness (unused, absorbed into convective coefficient)

    # Static args
    N, M = 20, 20
    args = {
      'dx': 0.1/N, 
      'dy': 0.1/M,
      'dt': 35.0,
      'tfinal': 400*60.0, 
      'N': 100,
      'M': 100,
      'D_T': alpha,
      'T_amb': 320,
      'h': 0.0,
      'num_save_points': 10,
    }

    if print_arguments:
        print("Arguments:")
        for key, value in args.items():
            print(f"{key}: {value}")

    # Initial conditions
    y0 = jax.numpy.ones((N,M), dtype=dtype)  * 320.0 # Human body temp
    cx, cy, dx, dy  = 10, 10, 1, 1 # Adding new ulcer
    q_src = jax.numpy.zeros((N,M), dtype=dtype)
    q_src = q_src.at[cx-dx:cx+dx, cy-dy:cy+dy].set(1.0)

    # Reverse mode
    q_src_gt = jax.numpy.array(q_src) 
    params = {'q_src': q_src_gt}

    print("\nSolving ground truth...")
    y = forward_solve(y0, args,params)

    # **************** INVERSE SOLVE ****************
    # Infer the Q source (static)

    # # Initial guess
    # key = jax.random.PRNGKey(0)
    # noise_level = 0.001
    # noise = noise_level * jax.random.normal(
    #         key,
    #         shape=q_src_gt.shape,
    #         dtype=q_src_gt.dtype,
    # )
    # q_src_pred = jax.numpy.array(q_src_gt) + noise
    
    # @eqx.filter_jit
    # def evaluate_loss(y0, args, params, y_gt):
    #   y = forward_solve(y0, args,params)
    #   return jax.numpy.mean((y-y_gt)**2)


    # # Optimizer
    # optimizer = optimistix.OptaxMinimiser(optax.adam(1e-3), rtol=1e-3, atol=1e-3)
    # step = equinox.filter_jit(equinox.Partial(solver.step, fn=fn, args=args))
    # state = optimizer.init(evaluate_loss, y0, args, options, f_struct, aux_struct, tags)


    
    # for i in tqdm.range(epochs):
    #     q_src_pred = params['q_src_pred']

    #     step(y=y, state=state)





    return y


if __name__ == "__main__":
    main()
