import argparse
import os
import time

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-{os.getuid()}")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
if not any(os.path.exists(path) for path in ("/dev/nvidia0", "/dev/nvidiactl")):
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print("Using CPU")
else:
    print("Using GPU")

import h5py
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import optax 


dtype = jnp.float32
try:
    device = jax.devices("cuda")[0]
except RuntimeError:
    device = jax.devices()[0]


@jax.jit
def laplacian(V, dx, dy):
    lap = jnp.zeros_like(V)
    lap = lap.at[1:-1, 1:-1].set(
        (V[2:, 1:-1] - 2 * V[1:-1, 1:-1] + V[:-2, 1:-1]) / dx**2
        + (V[1:-1, 2:] - 2 * V[1:-1, 1:-1] + V[1:-1, :-2]) / dy**2
    )
    lap = lap.at[0, 1:-1].set(
        2 * (V[1, 1:-1] - V[0, 1:-1]) / dx**2
        + (V[0, 2:] - 2 * V[0, 1:-1] + V[0, :-2]) / dy**2
    )
    lap = lap.at[-1, 1:-1].set(
        2 * (V[-2, 1:-1] - V[-1, 1:-1]) / dx**2
        + (V[-1, 2:] - 2 * V[-1, 1:-1] + V[-1, :-2]) / dy**2
    )
    lap = lap.at[1:-1, 0].set(
        (V[2:, 0] - 2 * V[1:-1, 0] + V[:-2, 0]) / dx**2
        + 2 * (V[1:-1, 1] - V[1:-1, 0]) / dy**2
    )
    lap = lap.at[1:-1, -1].set(
        (V[2:, -1] - 2 * V[1:-1, -1] + V[:-2, -1]) / dx**2
        + 2 * (V[1:-1, -2] - V[1:-1, -1]) / dy**2
    )
    lap = lap.at[0, 0].set(
        2 * (V[1, 0] - V[0, 0]) / dx**2 + 2 * (V[0, 1] - V[0, 0]) / dy**2
    )
    lap = lap.at[-1, -1].set(
        2 * (V[-2, -1] - V[-1, -1]) / dx**2
        + 2 * (V[-1, -2] - V[-1, -1]) / dy**2
    )
    lap = lap.at[0, -1].set(
        2 * (V[1, -1] - V[0, -1]) / dx**2
        + 2 * (V[0, -2] - V[0, -1]) / dy**2
    )
    lap = lap.at[-1, 0].set(
        2 * (V[-2, 0] - V[-1, 0]) / dx**2
        + 2 * (V[-1, 1] - V[-1, 0]) / dy**2
    )
    return lap


@jax.jit
def rhs(T, q_src, h, args):
    q_term = q_src / args["source_scale"]
    return (
        args["D_T"] * laplacian(T, args["dx"], args["dy"])
        + q_term
        - h * (T - args["T_amb"])
    )


def load_dataset(path):
    with h5py.File(path, "r") as f:
        T = np.asarray(f["T_history"], dtype=np.float32)
        t = np.asarray(f["time_vector"], dtype=np.float32).squeeze()
        x = np.asarray(f["x_vec"], dtype=np.float32).squeeze()
        y = np.asarray(f["y_vec"], dtype=np.float32).squeeze()
        q_history = np.asarray(f["q_history"], dtype=np.float32) if "q_history" in f else None

        params = {}
        if "params" in f:
            for key in f["params"]:
                params[key] = float(np.asarray(f["params"][key]).squeeze())

    valid = (t > 0.0) & np.isfinite(t) & np.isfinite(T).all(axis=(1, 2))
    order = np.argsort(t[valid])
    T = T[valid][order]
    t = t[valid][order] - t[valid][order][0]
    if q_history is not None:
        q_history = q_history[valid][order]
    return T, t, x, y, q_history, params


def make_args(T, t, x, y, params):
    rho = 1050.0
    C = 3770.0
    d_skin = 0.002
    rho_c = rho * C
    k = params.get("k", 0.6)
    return {
        "dx": float(np.mean(np.diff(x))),
        "dy": float(np.mean(np.diff(y))),
        "dt": float(np.mean(np.diff(t))),
        "tfinal": float(t[-1]),
        "N": T.shape[1],
        "M": T.shape[2],
        "D_T": k / rho_c,
        "T_amb": params.get("T_env_C", float(T[0].mean())),
        "h": 0.0,
        "source_scale": rho_c * d_skin,
    }


@jax.jit
def forward_solve(y0, q_src, h, dts, args):
    def step(T, dt):
        T_next = T + dt * rhs(T, q_src, h, args)
        return T_next, T_next

    _, ys = jax.lax.scan(step, y0, dts)
    return jnp.concatenate([y0[None, :, :], ys], axis=0)


@jax.jit
def q_compare_metrics(q_src, q_ref):
    diff = q_src - q_ref
    q_rmse = jnp.sqrt(jnp.mean(diff**2))
    q_mae = jnp.mean(jnp.abs(diff))
    q0 = q_src - jnp.mean(q_src)
    r0 = q_ref - jnp.mean(q_ref)
    q_corr = jnp.sum(q0 * r0) / (jnp.sqrt(jnp.sum(q0**2) * jnp.sum(r0**2)) + 1e-12)
    return q_rmse, q_mae, q_corr


def inverse_solve(
    T_data,
    t_data,
    args,
    epochs,
    lr,
    log_every,
    viz_every,
    smooth_weight,
    infer_h,
    initial_h,
    h_lr,
    max_h,
    q_history=None,
):
    solve_start = time.perf_counter()
    T_obs = jax.device_put(jnp.asarray(T_data, dtype=dtype), device)
    dts = jax.device_put(jnp.diff(jnp.asarray(t_data, dtype=dtype)), device)
    q_ref = None
    if q_history is not None:
        q_ref = jax.device_put(jnp.asarray(np.mean(q_history, axis=0), dtype=dtype), device)

    h0 = jnp.asarray(max(initial_h, 0.0), dtype=dtype)
    params_opt = {"q_src": jnp.zeros_like(T_obs[0]), "h": h0}
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params_opt)

    @jax.jit
    def loss_fn(params):
        pred = forward_solve(T_obs[0], params["q_src"], params["h"], dts, args)
        data_loss = jnp.mean((pred - T_obs) ** 2)
        q = params["q_src"]
        smooth = jnp.mean((q[1:, :] - q[:-1, :]) ** 2) + jnp.mean((q[:, 1:] - q[:, :-1]) ** 2)
        return data_loss + smooth_weight * smooth, (data_loss, pred)

    @jax.jit
    def train_step(params, state):
        (loss, (data_loss, pred)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        if not infer_h:
            grads = {**grads, "h": jnp.zeros_like(grads["h"])}
        updates, state = optimizer.update(grads, state, params)
        updates = {**updates, "h": updates["h"] * (h_lr / lr)}
        params = optax.apply_updates(params, updates)
        params = {**params, "h": jnp.clip(params["h"], 0.0, max_h)}
        if not infer_h:
            params = {**params, "h": h0}
        return params, state, loss, data_loss, pred

    history = []
    h_history = []
    q_metric_history = []
    snapshot_epochs = []
    q_snapshots = []
    h_snapshots = []
    pred_final_snapshots = []
    pred = None
    print(f"Using device: {device}")
    print(f"Fitting {T_obs.shape[0]} frames on a {T_obs.shape[1]}x{T_obs.shape[2]} grid")
    print(f"{'Inferring' if infer_h else 'Using fixed'} h with initial value {float(h0):.6e}")
    if infer_h:
        print(f"h learning rate {h_lr:.2e}, h clamp [0, {max_h:.2e}]")
    if q_ref is not None:
        print("q_history diagnostics use the time-mean ground truth and are not used by the optimizer")
    for epoch in range(1, epochs + 1):
        params_opt, opt_state, loss, data_loss, pred = train_step(params_opt, opt_state)
        q_src = params_opt["q_src"]
        h_value = params_opt["h"]
        take_snapshot = epoch == 1 or epoch % viz_every == 0 or epoch == epochs
        if take_snapshot:
            snapshot_epochs.append(epoch)
            q_snapshots.append(np.asarray(q_src))
            h_snapshots.append(float(h_value))
            pred_final_snapshots.append(np.asarray(pred[-1]))
        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            rmse = jnp.sqrt(data_loss)
            q_min = jnp.min(q_src)
            q_max = jnp.max(q_src)
            message = (
                f"epoch {epoch:5d} | loss {float(loss):.6e} | "
                f"rmse {float(rmse):.6f} C | q [{float(q_min):.4f}, {float(q_max):.4f}] | "
                f"h {float(h_value):.6e}"
            )
            if q_ref is not None:
                q_rmse, q_mae, q_corr = q_compare_metrics(q_src, q_ref)
                q_metric_history.append(
                    [epoch, float(q_rmse), float(q_mae), float(q_corr)]
                )
                message += (
                    f" | q_gt rmse {float(q_rmse):.4f} | "
                    f"mae {float(q_mae):.4f} | corr {float(q_corr):.4f}"
                )
            print(message)
        history.append(float(loss))
        h_history.append(float(h_value))

    snapshots = {
        "epochs": np.asarray(snapshot_epochs),
        "q_src": np.asarray(q_snapshots),
        "h": np.asarray(h_snapshots),
        "pred_final": np.asarray(pred_final_snapshots),
    }
    solve_seconds = time.perf_counter() - solve_start
    return (
        np.asarray(q_src),
        np.asarray(pred),
        np.asarray(history),
        np.asarray(h_history),
        np.asarray(q_metric_history),
        snapshots,
        solve_seconds,
    )


def animation_fps(num_frames, max_seconds):
    return max(1, int(np.ceil(num_frames / max_seconds)))


def animation_indices(num_frames, max_frames):
    if num_frames <= max_frames:
        return np.arange(num_frames)
    return np.unique(np.linspace(0, num_frames - 1, max_frames).astype(int))


def save_image_animation(
    path,
    images,
    epochs,
    title,
    cmap="coolwarm",
    center_zero=False,
    max_seconds=4.5,
):
    if len(images) == 0:
        return
    frames = images if len(images) > 1 else np.concatenate([images, images], axis=0)
    frame_epochs = epochs if len(epochs) > 1 else np.asarray([epochs[0], epochs[0]])
    vmin = float(np.min(frames))
    vmax = float(np.max(frames))
    if center_zero:
        vmax_abs = max(abs(vmin), abs(vmax))
        vmin, vmax = -vmax_abs, vmax_abs

    fig, ax = plt.subplots(figsize=(5, 4), dpi=120)
    im = ax.imshow(frames[0], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax)
    ax.set_title(f"{title}, epoch {int(frame_epochs[0])}")

    def update(i):
        im.set_data(frames[i])
        ax.set_title(f"{title}, epoch {int(frame_epochs[i])}")
        return (im,)

    ani = animation.FuncAnimation(fig, update, frames=len(frames), blit=False)
    ani.save(path, writer=animation.PillowWriter(fps=animation_fps(len(frames), max_seconds)))
    plt.close(fig)


def save_temperature_comparison_animation(
    path,
    T_data,
    pred,
    t_data,
    max_seconds=4.5,
    max_frames=120,
):
    if len(pred) == 0:
        return
    idx = animation_indices(len(pred), max_frames)
    measured = T_data[idx]
    reconstructed = pred[idx]
    residual = reconstructed - measured
    frame_times = t_data[idx]
    temp_vmin = float(min(np.min(measured), np.min(reconstructed)))
    temp_vmax = float(max(np.max(measured), np.max(reconstructed)))
    err_abs = float(np.max(np.abs(residual)))

    fig, axes = plt.subplots(1, 3, figsize=(10, 3), dpi=120)
    ims = [
        axes[0].imshow(measured[0], origin="lower", cmap="inferno", vmin=temp_vmin, vmax=temp_vmax),
        axes[1].imshow(reconstructed[0], origin="lower", cmap="inferno", vmin=temp_vmin, vmax=temp_vmax),
        axes[2].imshow(residual[0], origin="lower", cmap="coolwarm", vmin=-err_abs, vmax=err_abs),
    ]
    titles = ["Ground truth T", "Reconstructed T", "Reconstruction - GT"]
    for axis, im, title in zip(axes, ims, titles):
        axis.set_title(title)
        fig.colorbar(im, ax=axis, fraction=0.046)

    def update(i):
        ims[0].set_data(measured[i])
        ims[1].set_data(reconstructed[i])
        ims[2].set_data(residual[i])
        fig.suptitle(f"Temperature reconstruction, t={frame_times[i]:.1f}s")
        return ims

    update(0)
    fig.tight_layout()
    ani = animation.FuncAnimation(fig, update, frames=len(idx), blit=False)
    ani.save(path, writer=animation.PillowWriter(fps=animation_fps(len(idx), max_seconds)))
    plt.close(fig)


def save_q_triptych_animation(path, q_snapshots, q_ref, epochs, max_seconds=4.5):
    if len(q_snapshots) == 0:
        return
    frames = q_snapshots if len(q_snapshots) > 1 else np.concatenate([q_snapshots, q_snapshots], axis=0)
    frame_epochs = epochs if len(epochs) > 1 else np.asarray([epochs[0], epochs[0]])
    errors = frames - q_ref
    q_vmin = float(min(np.min(frames), np.min(q_ref)))
    q_vmax = float(max(np.max(frames), np.max(q_ref)))
    err_abs = float(np.max(np.abs(errors)))

    fig, axes = plt.subplots(1, 3, figsize=(10, 3), dpi=120)
    ims = [
        axes[0].imshow(frames[0], origin="lower", cmap="coolwarm", vmin=q_vmin, vmax=q_vmax),
        axes[1].imshow(q_ref, origin="lower", cmap="coolwarm", vmin=q_vmin, vmax=q_vmax),
        axes[2].imshow(errors[0], origin="lower", cmap="coolwarm", vmin=-err_abs, vmax=err_abs),
    ]
    titles = ["Inferred q", "Mean ground truth q", "Inferred - ground truth"]
    for axis, im, title in zip(axes, ims, titles):
        axis.set_title(title)
        fig.colorbar(im, ax=axis, fraction=0.046)

    def update(i):
        ims[0].set_data(frames[i])
        ims[2].set_data(errors[i])
        fig.suptitle(f"q convergence, epoch {int(frame_epochs[i])}")
        return ims

    update(0)
    fig.tight_layout()
    ani = animation.FuncAnimation(fig, update, frames=len(frames), blit=False)
    ani.save(path, writer=animation.PillowWriter(fps=animation_fps(len(frames), max_seconds)))
    plt.close(fig)


def metadata_h_ground_truth(params, override=None):
    if override is not None:
        return override
    for key in ("h", "h_gt", "h_conv", "h_coeff", "h_coefficient", "h0"):
        if key in params:
            return params[key]
    return None


def save_parameter_ratio_plot(
    path,
    snapshots,
    q_history,
    h_history,
    h_ground_truth=None,
    q_eps=1e-8,
):
    if snapshots is None or q_history is None or len(snapshots["epochs"]) == 0:
        return

    q_ref = np.mean(q_history, axis=0)
    valid = np.abs(q_ref) > q_eps
    if not np.any(valid):
        return

    q_ratio = snapshots["q_src"][:, valid] / q_ref[valid][None, :]
    epochs = snapshots["epochs"]
    colors = plt.cm.winter(np.linspace(0.15, 0.95, q_ratio.shape[1]))

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=180)
    for i in range(q_ratio.shape[1]):
        ax.plot(epochs, q_ratio[:, i], color=colors[i], alpha=0.25, linewidth=0.6)

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="ground truth")
    if h_ground_truth is not None and abs(h_ground_truth) > q_eps:
        h_ratio = np.asarray(h_history) / h_ground_truth
        ax.plot(
            np.arange(1, len(h_history) + 1),
            h_ratio,
            color="red",
            linewidth=2.0,
            label="h",
        )
    else:
        ax.text(
            0.01,
            0.02,
            "h ground truth unavailable",
            transform=ax.transAxes,
            color="red",
            fontsize=8,
            va="bottom",
        )

    finite = q_ratio[np.isfinite(q_ratio)]
    if finite.size:
        lo, hi = np.percentile(finite, [1, 99])
        margin = 0.1 * max(hi - lo, 1.0)
        ax.set_ylim(lo - margin, hi + margin)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Predicted / ground truth")
    ax.set_title("Parameter ratio convergence")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_outputs(
    out_dir,
    q_src,
    pred,
    T_data,
    t_data,
    history,
    h_history,
    q_history=None,
    q_metrics=None,
    snapshots=None,
    h_ground_truth=None,
    animation_seconds=4.5,
    animation_max_frames=120,
):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "inferred_heat_flux.npy"), q_src)
    np.save(os.path.join(out_dir, "pred_temperature.npy"), pred)
    np.savetxt(os.path.join(out_dir, "inferred_heat_flux.csv"), q_src, delimiter=",")
    np.savetxt(
        os.path.join(out_dir, "h_history.csv"),
        np.column_stack([np.arange(1, len(h_history) + 1), h_history]),
        delimiter=",",
        header="epoch,h",
        comments="",
    )

    fig, ax = plt.subplots(figsize=(5, 4), dpi=180)
    im = ax.imshow(q_src, origin="lower", cmap="coolwarm")
    fig.colorbar(im, ax=ax, label="Heat flux")
    ax.set_title("Inferred static heat flux")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "inferred_heat_flux.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3), dpi=180)
    ax.semilogy(history)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Inverse solve convergence")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "inverse_convergence.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3), dpi=180)
    ax.plot(np.arange(1, len(h_history) + 1), h_history)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("h")
    ax.set_title("Convective coefficient")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "h_history.png"))
    plt.close(fig)

    if q_metrics is not None and len(q_metrics) > 0:
        np.savetxt(
            os.path.join(out_dir, "q_ground_truth_metrics.csv"),
            q_metrics,
            delimiter=",",
            header="epoch,q_rmse,q_mae,q_corr",
            comments="",
        )
        fig, axes = plt.subplots(1, 2, figsize=(8, 3), dpi=180)
        axes[0].plot(q_metrics[:, 0], q_metrics[:, 1], label="RMSE")
        axes[0].plot(q_metrics[:, 0], q_metrics[:, 2], label="MAE")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Heat flux error")
        axes[0].legend()
        axes[1].plot(q_metrics[:, 0], q_metrics[:, 3])
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Correlation")
        axes[1].set_ylim(-1.05, 1.05)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "q_ground_truth_metrics.png"))
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3), dpi=180)
    vmin = min(float(T_data[-1].min()), float(pred[-1].min()))
    vmax = max(float(T_data[-1].max()), float(pred[-1].max()))
    for axis, image, title in zip(
        axes,
        [T_data[-1], pred[-1], pred[-1] - T_data[-1]],
        [f"Measured T, t={t_data[-1]:.1f}s", "Predicted T", "Residual"],
    ):
        limits = {"vmin": vmin, "vmax": vmax} if title != "Residual" else {}
        im = axis.imshow(image, origin="lower", cmap="inferno", **limits)
        fig.colorbar(im, ax=axis, fraction=0.046)
        axis.set_title(title)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "inverse_fit_final_frame.png"))
    plt.close(fig)

    save_temperature_comparison_animation(
        os.path.join(out_dir, "temperature_reconstruction_comparison.gif"),
        T_data,
        pred,
        t_data,
        max_seconds=animation_seconds,
        max_frames=animation_max_frames,
    )

    if q_history is not None:
        q_ref = np.mean(q_history, axis=0)
        fig, axes = plt.subplots(1, 3, figsize=(10, 3), dpi=180)
        for axis, image, title in zip(
            axes,
            [q_src, q_ref, q_src - q_ref],
            ["Inferred", "Mean dataset q_history", "Difference"],
        ):
            im = axis.imshow(image, origin="lower", cmap="coolwarm")
            fig.colorbar(im, ax=axis, fraction=0.046)
            axis.set_title(title)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "inverse_q_comparison.png"))
        plt.close(fig)

    if snapshots is not None and len(snapshots["epochs"]) > 0:
        np.savez(
            os.path.join(out_dir, "convergence_snapshots.npz"),
            epochs=snapshots["epochs"],
            q_src=snapshots["q_src"],
            h=snapshots["h"],
            pred_final=snapshots["pred_final"],
        )
        save_parameter_ratio_plot(
            os.path.join(out_dir, "parameter_ratio_convergence.png"),
            snapshots,
            q_history,
            h_history,
            h_ground_truth,
        )
        save_image_animation(
            os.path.join(out_dir, "q_src_convergence.gif"),
            snapshots["q_src"],
            snapshots["epochs"],
            "Inferred q",
            cmap="coolwarm",
            center_zero=True,
            max_seconds=animation_seconds,
        )
        save_image_animation(
            os.path.join(out_dir, "temperature_final_frame_convergence.gif"),
            snapshots["pred_final"],
            snapshots["epochs"],
            "Predicted final temperature",
            cmap="inferno",
            max_seconds=animation_seconds,
        )
        temp_residual = snapshots["pred_final"] - T_data[-1]
        save_image_animation(
            os.path.join(out_dir, "temperature_residual_convergence.gif"),
            temp_residual,
            snapshots["epochs"],
            "Final temperature residual",
            cmap="coolwarm",
            center_zero=True,
            max_seconds=animation_seconds,
        )
        if q_history is not None:
            q_ref = np.mean(q_history, axis=0)
            q_error = snapshots["q_src"] - q_ref
            save_image_animation(
                os.path.join(out_dir, "q_error_convergence.gif"),
                q_error,
                snapshots["epochs"],
                "q error",
                cmap="coolwarm",
                center_zero=True,
                max_seconds=animation_seconds,
            )
            save_image_animation(
                os.path.join(out_dir, "q_abs_error_convergence.gif"),
                np.abs(q_error),
                snapshots["epochs"],
                "Absolute q error",
                cmap="magma",
                max_seconds=animation_seconds,
            )
            save_q_triptych_animation(
                os.path.join(out_dir, "q_ground_truth_triptych_convergence.gif"),
                snapshots["q_src"],
                q_ref,
                snapshots["epochs"],
                max_seconds=animation_seconds,
            )


def main():
    total_start = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/ulcer_dataset_01.mat")
    parser.add_argument("--out-dir", default="outputs/ulcer_dataset_01_inverse")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--viz-every", type=int, default=None)
    parser.add_argument("--animation-seconds", type=float, default=4.5)
    parser.add_argument("--animation-max-frames", type=int, default=120)
    parser.add_argument("--smooth-weight", type=float, default=1e-4)
    parser.add_argument("--initial-h", type=float, default=0.0)
    parser.add_argument("--h-lr", type=float, default=1e-6)
    parser.add_argument("--max-h", type=float, default=1e-3)
    parser.add_argument("--h-ground-truth", type=float, default=None)
    parser.add_argument("--no-infer-h", action="store_true")
    args_cli = parser.parse_args()
    viz_every = args_cli.viz_every or args_cli.log_every

    load_start = time.perf_counter()
    T_data, t_data, x, y, q_history, params = load_dataset(args_cli.dataset)
    load_seconds = time.perf_counter() - load_start
    args = make_args(T_data, t_data, x, y, params)
    h_ground_truth = metadata_h_ground_truth(params, args_cli.h_ground_truth)

    print("Arguments:")
    for key, value in args.items():
        print(f"{key}: {value}")

    q_src, pred, history, h_history, q_metrics, snapshots, solve_seconds = inverse_solve(
        T_data,
        t_data,
        args,
        args_cli.epochs,
        args_cli.lr,
        args_cli.log_every,
        viz_every,
        args_cli.smooth_weight,
        not args_cli.no_infer_h,
        args_cli.initial_h,
        args_cli.h_lr,
        args_cli.max_h,
        q_history,
    )
    save_start = time.perf_counter()
    save_outputs(
        args_cli.out_dir,
        q_src,
        pred,
        T_data,
        t_data,
        history,
        h_history,
        q_history,
        q_metrics,
        snapshots,
        h_ground_truth=h_ground_truth,
        animation_seconds=args_cli.animation_seconds,
        animation_max_frames=args_cli.animation_max_frames,
    )
    save_seconds = time.perf_counter() - save_start
    total_seconds = time.perf_counter() - total_start

    print(f"\nFinal loss: {history[-1]:.6e}")
    print(f"Final heat flux min/max: {q_src.min():.6f}, {q_src.max():.6f}")
    print(f"Final h: {h_history[-1]:.6e}")
    print(f"Saved results to {args_cli.out_dir}")
    print("\nTiming:")
    print(f"  dataset load: {load_seconds:.2f} s")
    print(f"  inverse solve: {solve_seconds:.2f} s")
    print(f"  avg per epoch: {solve_seconds / args_cli.epochs:.4f} s")
    print(f"  output + animations: {save_seconds:.2f} s")
    print(f"  total: {total_seconds:.2f} s")


if __name__ == "__main__":
    main()
