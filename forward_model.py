import torch
from torch.autograd import forward_ad
import matplotlib
import matplotlib.pyplot as plt
import os
import tqdm

# Setting GPU
dtype = torch.float
device = torch.device("cuda")

@torch.compile
def laplacian(V,dx,dy):
    lap = torch.zeros_like(V)
    lap[1:-1, 1:-1] = (V[2:, 1:-1] - 2 * V[1:-1, 1:-1] + V[:-2, 1:-1]) / dx**2.0 + \
      (V[1:-1, 2:] - 2 * V[1:-1, 1:-1] + V[1:-1,:-2]) / dy**2.0

    # X boundaries
    lap[0, 1:-1] = 2*(V[1, 1:-1]- V[0, 1:-1]) / dx**2 + (V[0, 2:] - 2 * V[0, 1:-1] + V[0,:-2]) / dy**2.0
    lap[-1, 1:-1] = 2*(-V[-1, 1:-1]+ V[-2, 1:-1]) / dx**2 + (V[-1, 2:] - 2 * V[-1, 1:-1] + V[-1,:-2]) / dy**2.0

    # Y boundaries
    lap[1:-1, 0] = (V[2:, 0] - 2 * V[1:-1, 0] + V[:-2, 0]) / dx**2.0 + 2*(V[1:-1, 1]- V[1:-1, 0]) / dy**2 
    lap[1:-1, -1] = (V[2:, -1] - 2 * V[1:-1, -1] + V[:-2, -1]) / dx**2.0+ 2*(-V[1:-1,-1]+ V[ 1:-1,-2]) / dy**2

    # Corners
    lap[0,0] = 2 * (V[1, 0] - V[0, 0]) / dx**2.0 + 2 * (V[0, 1] - V[0, 0]) / dy **2
    lap[-1,-1] = 2 * (V[-2, -1] - V[-1, -1]) / dx**2.0 + 2 * (V[-1, -2] - V[-1, -1]) / dy **2
    lap[0,-1] = 2 * (V[1, -1] - V[0, -1]) / dx**2.0 + 2 * (V[0, -2] - V[0, -1]) / dy **2
    lap[-1,0] = 2 * (V[-2, 0] - V[-1, 0]) / dx**2.0 + 2 * (V[-1, 1] - V[-1, 0]) / dy **2
    return lap


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
    plot_gt_interval = args['plot_gt_interval']

    t0 = 0.0
    tfinal = args['tfinal']
    nsteps = int((tfinal - t0)/dt)
    dydt = torch.ones(N,M,2, dtype=dtype, device=device)
    fig,ax = plt.subplots(1,1,dpi=250)

    y = y0
    print("\nBeginning timesteps")
    for step in tqdm.tqdm(range(nsteps), desc="Time stepping"):      
      t = t0 + step * dt                                  
      dy_dt = rhs(t, y, dydt, args, params)
      y = y + dt * dy_dt


      # plotting
      if step % plot_gt_interval == 0:
        im0 = ax.imshow(y[:,:].detach().cpu().numpy())#, vmin=300, vmax=350)
        if t == dt:
          fig.colorbar(im0, ax=ax)
        plt.savefig(f"ulcer_{t}.png")



def main():

    # Output directori
    #os.mkdir("data", exist_ok=True)
    plot_gt_interval = 1000
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
      'plot_gt_interval': plot_gt_interval
    }
    dt = min(args['dx']**2, args['dy']**2) / (2 * args['D_T']) # Stability requirement: dt <= dx^2 / (4 * D)

    if print_arguments:
        print("Arguments:")
        for key, value in args.items():
            print(f"{key}: {value}")
        print(f"dt: {dt}")

    # Initial conditions
    y0 = torch.ones(N,M, dtype=dtype, device=device)  * 310.0 # Human body temp
    cx, cy, dx,dy  = 50,50, 4,4 # Adding new ulcer
    q_src = torch.zeros((N,M), dtype=dtype, device=device)
    q_src[cx-dx:cx+dx, cy-dy:cy+dy] = 10.0

    # Reverse mode
    q_src = torch.nn.Parameter(q_src) 
    params = {'q_src': q_src}

    print("\nSolving ground truth...")
    y = forward_solve(dt, y0, args,params)

    return y


if __name__ == "__main__":
    main()
