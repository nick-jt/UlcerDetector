import torch
import matplotlib
import matplotlib.pyplot as plt
import os

def rhs(t, y, dydt, args):
    """2-D reaction-diffusion model of skin
    
    dT(x,y,t)/dt = D_T(x,y) * laplacian(T) + f(T, U)
    dU(x,y,t)/dt = g(T, U)
    
    for T = temperature and U = ulcer value
        x,y = positions
        t = time
        D_T(x,y) = diffusion coefficient of temperature
        f(T, U) = reaction term for temperature
        g(T, U) = reaction term for ulcer
    
    Args:
        t: time
        y: state vector [N, M, 2]
            y[:, :, 0] = temperature
            y[:, :, 1] = ulcer value
    
    Returns:
        rhs: right-hand side of the ODE
    """
    T, U = y[:,:,0], y[:,:,1]
    D_T = 1000.0
    f = lambda T, U: 0.0
    g = lambda T, U: 0.0
    dx = args['dx']
    dy = args['dy']

    def laplacian(V,dx,dy):
        lap = torch.zeros_like(V)
        lap[1:-1, 1:-1] = (V[2:, 1:-1] - 2 * V[1:-1, 1:-1] + V[:-2, 1:-1]) / dx**2.0 + (V[1:-1, 2:] - 2 * V[1:-1, 1:-1] + V[1:-1,:-2]) / dy**2.0

        # X boundaries
        lap[0, 1:-1] = 2*(V[1, 1:-1]- V[0, 1:-1]) / dx**2 + (V[0, 2:] - 2 * V[0, 1:-1] + V[0,:-2]) / dy**2.0
        lap[-1, 1:-1] = 2*(-V[-1, 1:-1]+ V[-2, 1:-1]) / dx**2 + (V[-1, 2:] - 2 * V[-1, 1:-1] + V[-1,:-2]) / dy**2.0
        # Y boundaries
        lap[1:-1, 0] = (V[2:, 0] - 2 * V[1:-1, 0] + V[:-2, 0]) / dx**2.0 + 2*(V[1:-1, 1]- V[1:-1, 0]) / dy**2 
        lap[1:-1, -1] = (V[2:, -1] - 2 * V[1:-1, -1] + V[:-2, -1]) / dx**2.0+ 2*(-V[1:-1,-1]+ V[ 1:-1,-2]) / dy**2 
        # Corners
        lap[0,0] = # TODO
        return lap

    laplacian_T = laplacian(T, dx,dy)

    dydt[:,:,0] = D_T * laplacian_T + f(T, U)
    dydt[:,:,1] = g(T, U)

    return dydt



def main():

    # Output directori
    #os.mkdir("data", exist_ok=True)
    plot_gt = True



    # Run forward model
    N, M = 100, 100
    t = 0.0
    dt = 0.1
    args = {'dx': 1.0/N, 'dy':1.0/M}
    y = torch.ones(N,M,2) 
    y[:,:,0] *= 310.0 # average human body temp
    y[:,:,1] *= 0.0 # no ulcers anywhere
    
    # Ground truth ulcer
    cx, cy, dx,dy  = 50,50, 4,4
    y[cx-dx:cx+dx, cy-dy:cy+dy, 0] = 320
    y[cx-dx:cx+dx, cy-dy:cy+dy, 1] = 1


    dydt = torch.ones(N,M,2)
    fig,ax = plt.subplots(1,2,dpi=250)

    while t < 1.0:
        dy_dt = rhs(t, y, dydt,args)
        y = y + dt * dy_dt
        t += dt

        # plotting
        if plot_gt:
          im0 = ax[0].imshow(y[:,:,0].detach().cpu().numpy(), vmin=300, vmax=350)
          im1 = ax[1].imshow(y[:,:,1].detach().cpu().numpy(), vmin=0, vmax=1)
          if t == dt:
            fig.colorbar(im0, ax=ax[0])
            fig.colorbar(im1, ax=ax[1])
          plt.savefig(f"ulcer_{t}.png")


    return y





if __name__ == "__main__":
    main()
