import matplotlib.pyplot as plt
import numpy as np

from src.utils import load_degree_distribution


def dist_plot(
    x,
    y,
    *,
    ax=None,
    log_scale: bool = (False, False),
    title: str = "Degree Distribution",
    xlabel: str = "Degree",
    ylabel: str = "Probability",
    marker: str = "o",
    linestyle: str = "None",
    grid_axis: bool = "both",
):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    if log_scale[0]:
        ax.set_xscale("log")
    if log_scale[1]:
        ax.set_yscale("log")
    if log_scale[0] and log_scale[1]:
        ax.set_aspect("equal", adjustable="box")

    # ax.autoscale(enable=True, axis="both", tight=True)
    ax.plot(x, y, marker=marker, linestyle=linestyle)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid(True, which=grid_axis, alpha=0.3)

    ax.set_box_aspect(1)
    return ax


def overlay_model(ax, model, kmax, **plot_kwargs):
    ks, pmf = model.pmf_over_degrees(kmax)
    ax.plot(ks, pmf, scalex=False, scaley=False, clip_on=True, **plot_kwargs)
    return ax


def dist_plot_render_one(lang, perspective, seqdir, outdir):
    from src.utils import LANG_DICT
    import os
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dist = load_degree_distribution(seqdir, lang)
    degrees = np.array(sorted(dist.keys()))
    counts = np.array([dist[k] for k in degrees])
    probs = counts / counts.sum()
    y = counts if perspective == "Counts" else probs

    fig, axes = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)

    for xi in (0, 1):
        for yi in (0, 1):
            xlog, ylog = bool(xi), bool(yi)
            title = f"{perspective} - x: {'log' if xlog else 'linear'}, y: {'log' if ylog else 'linear'}"
            dist_plot(
                degrees,
                y,
                ax=axes[xi, yi],
                log_scale=(xlog, ylog),
                title=title,
                ylabel=("Count" if perspective == "Counts" else "P(K = k)"),
            )

    fig.suptitle(f"Degree Distribution for {LANG_DICT[lang].capitalize()}", fontsize=16)
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{LANG_DICT[lang]}_{perspective}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def model_fit_render_one(lang, fitted_models, seqdir, outdir):
    from src.utils import LANG_DICT
    import matplotlib
    import os

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dist = load_degree_distribution(seqdir, lang)
    degrees = np.array(sorted(dist.keys()))
    counts = np.array([dist[k] for k in degrees])
    probs = counts / counts.sum()

    models = list(fitted_models.values())
    n = len(models)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8), constrained_layout=True)

    for i in [0, 1]:
        for ax, model in zip(axes[i], models):
            dist_plot(
                degrees,
                probs,
                ax=ax,
                log_scale=(bool(i), bool(i)),
                title=f"{model.__class__.__name__} Fit",
                ylabel="P(K = k)",
            )

            overlay_model(
                ax,
                model,
                model.stats.kmax_obs,
                label=model.__class__.__name__,
                linestyle="-",
            )
            ax.legend()

    xlim_log = (max(1, degrees.min()), degrees.max())
    ymin = np.nanmin(probs[probs > 0])
    ymax = np.nanmax(probs)
    ylim_log = (ymin * 0.9, ymax * 1.1)
    for ax in axes[1]:
        ax.set_xlim(xlim_log)
        ax.set_ylim(ylim_log)

    fig.suptitle(f"Model Fit for {LANG_DICT[lang].capitalize()}", fontsize=16)
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{LANG_DICT[lang]}_model_fit.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _render_model_fit_core(degrees, probs, models, title, basename, outdir):
    import os
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(models)

    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8), constrained_layout=True)

    for i in (0, 1):  # row 0: linear-linear, row 1: log-log
        for ax, model in zip(axes[i], models):
            dist_plot(
                degrees,
                probs,
                ax=ax,
                log_scale=(bool(i), bool(i)),
                title=f"{type(model).__name__} Fit",
                ylabel="P(K = k)",
            )
            overlay_model(
                ax,
                model,
                model.stats.kmax_obs,
                label=type(model).__name__,
                linestyle="-",
            )
            ax.legend()

    xlim_log = (max(1, degrees.min()), degrees.max())
    ymin = np.nanmin(probs[probs > 0])
    ymax = np.nanmax(probs)
    ylim_log = (ymin * 0.9, ymax * 1.1)
    for ax in axes[1]:
        ax.set_xlim(xlim_log)
        ax.set_ylim(ylim_log)

    fig.suptitle(title, fontsize=16)
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{basename}_model_fit.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def model_fit_render_dataset(ds_id, models_dict, samples_dir, outdir):
    from src.utils import load_degree_sequence
    import numpy as np

    dist_tag, p = ds_id.split("_", 1)
    sample_path = f"{samples_dir}/sample_of_{dist_tag}_with_parameter_{p}.txt"

    deg_list = load_degree_sequence(fixed_path=sample_path)
    ks, counts = np.unique(np.asarray(deg_list, dtype=int), return_counts=True)
    ks = ks[ks >= 1]
    counts = counts[-len(ks) :] if counts.size != ks.size else counts
    probs = counts / counts.sum()

    models = [models_dict[i] for i in sorted(models_dict.keys())]
    title = f"Model Fit for {ds_id}"
    basename = ds_id
    return _render_model_fit_core(ks, probs, models, title, basename, outdir)


def model_fit_render_one(lang, fitted_models, seqdir, outdir):
    from src.utils import LANG_DICT, load_degree_distribution
    import numpy as np

    dist = load_degree_distribution(seqdir, lang)
    degrees = np.array(sorted(dist.keys()))
    counts = np.array([dist[k] for k in degrees])
    probs = counts / counts.sum()
    models = list(fitted_models.values())

    title = f"Model Fit for {LANG_DICT[lang].capitalize()}"
    basename = f"{LANG_DICT[lang]}"
    return _render_model_fit_core(degrees, probs, models, title, basename, outdir)
