"""
Plotting
"""

import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import gc
import jax.numpy as jnp
import copy

plot_format = "png"

COLUMN_WIDTH_INCHES = 3.464


def set_plot_style():
    """
    Sets the plotting style. Important to unify things across figures and machines and generally makes plots prettier.
    Thank you to Federico Bianchini for this template!
    """
    rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
    plt.rcParams.update({"text.usetex": True, "font.family": "serif"})
    # rc("text", usetex=True)
    # rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    # plt.rcParams['font.family'] = 'sans-serif'
    # plt.rcParams['font.sans-serif'] = ['Tahoma']
    # plt.rcParams['mathtext.fontset'] = 'cm'
    # rc('font',**{'family':'sans-serif'})#,'dejavuserif':['Computer Modern']})
    # plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams["axes.linewidth"] = 1.5
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["xtick.labelsize"] = 8  # 12
    plt.rcParams["ytick.labelsize"] = 8  # 12
    plt.rcParams["xtick.major.size"] = 4  # 7
    plt.rcParams["ytick.major.size"] = 4  # 7
    plt.rcParams["xtick.minor.size"] = 2  # 4
    plt.rcParams["ytick.minor.size"] = 2  # 4
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["legend.frameon"] = False

    plt.rcParams["xtick.major.width"] = 1
    plt.rcParams["ytick.major.width"] = 1
    plt.rcParams["xtick.minor.width"] = 1
    plt.rcParams["ytick.minor.width"] = 1
    plt.clf()
    plt.close()
    # sns.set(rc('font',**{'family':'serif','serif':['Computer Modern']}))
    # sns.set_style("ticks", {'figure.facecolor': 'grey'})


def covmat_diagonal_plot(covmat, title, file_name):
    if np.max(np.abs(np.diag(covmat))) == 0.0:
        return
    plt.figure()
    plt.grid()
    plt.title(title)
    plt.plot(np.abs(np.diag(covmat)))
    plt.xlabel("Data bin")
    plt.ylabel("Diagonal Covariance")
    plt.yscale("log")
    plt.savefig(file_name)
    plt.close()

    gc.collect()


def data_covmat_plot(covmat, title, file_name, log=True):
    plt.figure()
    plt.axis("equal")
    plt.title(title)
    loc_covmat = jnp.asarray(covmat)
    max_abs = np.max(np.abs(loc_covmat))
    if max_abs == 0.0:
        plt.close()
        return
    min_value = np.min(np.abs(loc_covmat))
    loc_covmat = loc_covmat.at[loc_covmat == 0.0].set(min_value)
    if log:
        plt.imshow(jnp.abs(loc_covmat), cmap="jet", norm="log")
    else:
        plt.imshow(loc_covmat, cmap="seismic", vmin=-max_abs, vmax=max_abs)
    plt.colorbar()
    plt.savefig(file_name)
    plt.close()
    gc.collect()


def variance_plots(variances, title, y_label, file_name):
    plt.figure()
    plt.grid()
    plt.title(title)
    components = np.arange(len(variances)) + 1
    plt.plot(components, variances)
    plt.xlabel("PCA component")
    plt.ylabel(y_label)
    plt.yscale("log")
    plt.savefig(file_name)
    plt.close()

    gc.collect()


def eigenvector_plots(eigenvectors, title, file_name):
    plt.figure()
    plt.grid()
    plt.title(title)
    x = np.arange(len(eigenvectors[0])) + 1
    for i in range(len(eigenvectors)):
        plt.plot(x, eigenvectors[i], label="Component " + str(i))
    plt.legend()
    plt.xlabel("data bin")
    plt.ylabel("Eigenvector")
    plt.savefig(file_name)
    plt.close()

    gc.collect()


def plot_loglikes(loglikes, parameters, xlabel, file_name):
    plt.figure()
    plt.grid()
    plt.scatter(parameters, loglikes)
    plt.xlabel(xlabel)
    plt.ylabel("Loglike")
    plt.savefig(file_name)
    plt.close()

    gc.collect()


E_kin_energygrid = np.array(
    [
        1.000000e00,
        1.400000e00,
        1.960000e00,
        2.744000e00,
        3.841600e00,
        5.378240e00,
        7.529536e00,
        1.054135e01,
        1.475789e01,
        2.066105e01,
        2.892547e01,
        4.049565e01,
        5.669391e01,
        7.937148e01,
        1.111201e02,
        1.555681e02,
        2.177953e02,
        3.049135e02,
        4.268789e02,
        5.976304e02,
        8.366826e02,
        1.171356e03,
        1.639898e03,
        2.295857e03,
        3.214200e03,
        4.499880e03,
        6.299831e03,
        8.819764e03,
        1.234767e04,
        1.728674e04,
        2.420143e04,
        3.388201e04,
        4.743481e04,
        6.640873e04,
        9.297222e04,
        1.301611e05,
        1.822256e05,
        2.551158e05,
        3.571621e05,
        5.000269e05,
        7.000377e05,
        9.800528e05,
        1.372074e06,
        1.920903e06,
        2.689265e06,
        3.764971e06,
        5.270959e06,
        7.379343e06,
        1.033108e07,
        1.446351e07,
        2.024892e07,
        2.834848e07,
        3.968788e07,
        5.556303e07,
        7.778824e07,
        1.089035e08,
        1.524649e08,
        2.134509e08,
        2.988313e08,
        4.183638e08,
        5.857093e08,
        8.199931e08,
        1.147990e09,
    ]
)


def plot_normalization(X, data, label):

    norm_factor = jnp.ones(len(data[0]))

    if label in ["tt", "te", "ee", "bb", "tb", "eb"]:
        norm_factor = X * (X + 1)
    if label in ["pp", "tp"]:
        norm_factor = X * X * (X + 1) * (X + 1)
    if label in ["proton_unmodulated"]:
        norm_factor = 1  # X*X

    return norm_factor


def plain_plot(x, y, label, file_name):
    plt.figure()
    plt.plot(x, y, label=label)
    plt.legend()
    plt.savefig(file_name)
    plt.close()

    gc.collect()


def error_plot(x, y, std, file_name):
    # plot a slice of the trained GP
    plt.figure()
    plt.plot(x, y)
    plt.fill_between(x, y - 3.0 * std, y + 3.0 * std, alpha=0.3)
    plt.fill_between(x, y - std, y + std, alpha=0.3)
    plt.savefig(file_name)
    plt.close()

    gc.collect()


def plain_scatter(x, y, label, file_name):
    plt.figure()
    plt.scatter(x, y, label=label)
    plt.legend()
    plt.savefig(file_name)
    plt.close()

    gc.collect()


def data_plot_raw(data, label, file_name):
    if data.shape[1] == 1:
        # just scatter the data
        plt.figure()
        plt.grid()
        plt.scatter(range(len(data)), data)
        plt.xlabel("Data index")
        plt.ylabel(label)
        plt.savefig(file_name)
        plt.close()

    else:
        N = len(data)
        X = np.array(range(len(data[0])))

        norm_factor = plot_normalization(X, data, label)

        plt.figure()
        plt.grid()
        for i in range(N):
            plt.plot(X, data[i] * norm_factor)
        plt.xlabel("Data index")
        plt.ylabel(label)
        plt.savefig(file_name)
        plt.close()

    gc.collect()


def data_plot_normalized(data, label, file_name):
    if data.shape[1] == 1:
        # just scatter the data
        plt.figure()
        plt.grid()
        plt.scatter(range(len(data)), data)
        plt.xlabel("Data index")
        plt.ylabel("Normalized " + label)
        plt.savefig(file_name)
        plt.close()

    else:
        N = len(data)
        X = np.array(range(len(data[0])))
        plt.figure()
        plt.grid()
        for i in range(N):
            plt.plot(X, data[i])
        plt.xlabel("Data index")
        plt.ylabel(label)
        plt.savefig(file_name)
        plt.close()

    gc.collect()


def pca_parameter_plot(x, y, x_label, y_label, title, file_name):
    plt.figure()
    plt.grid()
    plt.title(title)
    plt.scatter(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(file_name)
    plt.close()

    gc.collect()


def loss_plot(loss, title, file_name):
    plt.figure()
    plt.grid()
    plt.title(title)
    epoch = range(len(loss))
    plt.plot(epoch, loss)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig(file_name)
    plt.close()

    gc.collect()


def plot_pca_components_test_set(true, pred, pred_std, err_tol, title, file_name):
    # plot residuals
    plt.figure()
    plt.grid()
    plt.title(title)
    plt.errorbar(
        true, true - pred, yerr=pred_std, fmt="o", label="Prediction and total error"
    )
    plt.errorbar(true, true - pred, yerr=err_tol, fmt="o", label="White Noise Level")
    plt.legend()
    plt.xlabel("True")
    plt.ylabel("Residuals")
    plt.savefig(file_name)
    plt.close()

    gc.collect()


def plot_prediction_test(
    prediction, true, std, err_tol, title, data_point, file_name, data_covmat
):

    # create mask where data_covmat is zero
    if data_covmat is None:
        mask = jnp.ones(len(true[0]))
    else:
        mask = jnp.where(jnp.diag(data_covmat) == 0.0, 0.0, 1.0)

    X = np.array(range(len(true[0])))
    norm_factor = plot_normalization(X, true, title)

    # make 2 subplots
    fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(10, 20))
    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    ax[0].set_title(title + " \n ")

    # add text with str(data_point) into upper left corner
    ax[0].text(
        0.05,
        0.95,
        "Data point: " + str(data_point),
        ha="left",
        va="bottom",
        transform=plt.gca().transAxes,
    )

    if len(prediction) == 1:
        # plot masked prediction
        ax[0].errorbar(
            [0],
            prediction * norm_factor * mask,
            yerr=std * mask,
            fmt="o",
            label="Masked Prediction",
        )
        ax[0].errorbar(
            [0],
            prediction * norm_factor * mask,
            yerr=err_tol * mask,
            fmt="o",
            label="White Noise",
        )

        ax[0].errorbar([0], true * norm_factor, fmt="o", label="True")

        # plot masked residuals
        ax[1].errorbar(
            [0],
            (true - prediction) * norm_factor * mask,
            yerr=std * norm_factor * mask,
            fmt="o",
            label="Masked Residuals",
        )
        ax[1].errorbar(
            [0],
            (true - prediction) * norm_factor * mask,
            yerr=err_tol * norm_factor * mask,
            fmt="o",
            label="White Noise",
        )

        # plot masked residuals
        ax[2].errorbar(
            [0],
            (true - prediction) / std * mask,
            yerr=std / std * mask,
            fmt="o",
            label="Masked Residuals",
        )
        ax[2].errorbar(
            [0],
            (true - prediction) / std * mask,
            yerr=err_tol / std * mask,
            fmt="o",
            label="White Noise",
        )

    else:
        # plot masked prediction
        ax[0].plot(
            range(len(true[0])), prediction * norm_factor * mask, label="Prediction"
        )

        # make errorband around the prediction
        ax[0].fill_between(
            range(len(true[0])),
            (prediction - std) * norm_factor * mask,
            (prediction + std) * norm_factor * mask,
            alpha=0.5,
            label="1$\sigma$",
        )
        ax[0].fill_between(
            range(len(true[0])),
            (prediction - err_tol) * norm_factor * mask,
            (prediction + err_tol) * norm_factor * mask,
            alpha=0.5,
            label="White Noise",
        )
        ax[0].plot(range(len(true[0])), true[0] * norm_factor, label="True")

        # make residuals
        ax[1].plot(
            range(len(true[0])),
            (true[0] - prediction) * norm_factor * mask,
            label="Residuals",
        )
        ax[1].fill_between(
            range(len(true[0])),
            -std * norm_factor * mask,
            std * norm_factor * mask,
            alpha=0.5,
            label="1$\sigma$",
        )
        ax[1].fill_between(
            range(len(true[0])),
            -err_tol * norm_factor * mask,
            err_tol * norm_factor * mask,
            alpha=0.5,
            label="White Noise",
        )

        # make residuals
        ax[2].plot(
            range(len(true[0])), (true[0] - prediction) * mask / std, label="Residuals"
        )
        ax[2].fill_between(range(len(true[0])), -1, 1, alpha=0.5, label="1$\sigma$")
        ax[2].fill_between(
            range(len(true[0])),
            -err_tol / std,
            err_tol / std,
            alpha=0.5,
            label="White Noise",
        )

    ax[2].set_xlabel("Data index")
    ax[0].set_ylabel("Prediction")
    ax[1].set_ylabel("Prediction - True")
    ax[2].set_ylabel("(Prediction - True) / $\sigma$")
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    plt.savefig(file_name)
    plt.close()

    gc.collect()
