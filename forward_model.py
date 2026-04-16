import jax 
from jax import grad, jit, vmap, pmap, value_and_grad, jacfwd, jacrev, hessian
import matplotlib
import matplotlib.pyplot as plt
import os
import tqdm
import diffrax

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
def rhs(t, y, dydt, args, params):
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
    dydt = jax.numpy.ones((N,M), dtype=dtype)
    fig, ax = plt.subplots(1, 1, dpi=250)

    print("\nBeginning timesteps")
    # for step in tqdm.tqdm(range(nsteps), desc="Time stepping"):      
    #   t = t0 + step * dt                                  
    #   dy_dt = rhs(t, y, dydt, args, params)
    #   y = y + dt * dy_dt

    def rhs_fn_wrapper(t, y, args):
        return rhs(t, y, dydt, args, params)

    term = diffrax.ODETerm(rhs_fn_wrapper)
    rtol = 1e-3
    atol = 1e-3
    stepsize_controller = diffrax.PIDController(
        pcoeff=0.3, icoeff=0.4, rtol=rtol, atol=atol, dtmax=0.01
    )
    save_ts = jax.numpy.linspace(t0, tfinal, num_save_points)
    saveat = diffrax.SaveAt(ts=save_ts)
    progress_meter = diffrax.TqdmProgressMeter()
    sol = diffrax.diffeqsolve(
        term, 
        solver=diffrax.Heun(), 
        t0=t0,
        t1=tfinal,
        dt0=dt,
        y0=y0,
        args=args,
        max_steps=None,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        progress_meter=progress_meter
    )

    ys = jax.numpy.asarray(sol.ys)
    vmin, vmax = float(ys.min()), float(ys.max())
    for i in range(len(sol.ts)):
        ax.clear()
        im0 = ax.imshow(ys[i], vmin=vmin, vmax=vmax)
        if i == 0:
            ax.set_title("Initial Condition")
            ax.set_title("Final Condition")
            ax.set_title(f"Time: {sol.ts[i]}")
            fig.colorbar(im0, ax=ax)
        plt.savefig(f"ulcer_{i}.png")

    return sol


def main():

    # Output directori
    #os.mkdir("data", exist_ok=True)
    print_arguments = True

    # Static args
    N, M = 100, 100
    args = {
      'dx': 1.0/N, 
      'dy': 1.0/M,
      'tfinal': 1.0,
      'N': 100,
      'M': 100,
      'D_T': 10.0,
      'T_amb': 320,
      'h': 0.0,
      'num_save_points': 10,
    }
    dt = 0.01 #min(args['dx']**2, args['dy']**2) / (4 * args['D_T']) # Stability requirement: dt <= dx^2 / (4 * D)

    if print_arguments:
        print("Arguments:")
        for key, value in args.items():
            print(f"{key}: {value}")
        print(f"dt: {dt}")

    # Initial conditions
    y0 = jax.numpy.ones((N,M), dtype=dtype)  * 310.0 # Human body temp
    cx, cy, dx,dy  = 50,50, 4,4 # Adding new ulcer
    q_src = jax.numpy.zeros((N,M), dtype=dtype)
    q_src = q_src.at[cx-dx:cx+dx, cy-dy:cy+dy].set(10.0)

    # Reverse mode
    q_src = jax.numpy.array(q_src) 
    params = {'q_src': q_src}

    print("\nSolving ground truth...")
    y = forward_solve(dt, y0, args,params)

    return y


if __name__ == "__main__":
    main()
