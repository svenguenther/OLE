import matplotlib.pyplot as plt
import numpy as np
import gc
import jax.numpy as jnp
import copy

plot_format = 'png'

def covmat_diagonal_plot(covmat, title, file_name):
    if np.max(np.abs(np.diag(covmat))) == 0.0:
        return
    plt.figure()
    plt.grid()
    plt.title(title)
    plt.plot(np.abs(np.diag(covmat)))
    plt.xlabel('Data bin')
    plt.ylabel('Diagonal Covariance')
    plt.yscale('log')
    plt.savefig(file_name)
    plt.close()

    gc.collect()

def data_covmat_plot(covmat, title, file_name, log=True):
    plt.figure()
    plt.axis('equal')
    plt.title(title)
    loc_covmat = jnp.asarray(covmat)
    max_abs = np.max(np.abs(loc_covmat))
    if max_abs == 0.0:
        plt.close()
        return 
    min_value = np.min(np.abs(loc_covmat))
    loc_covmat =loc_covmat.at[loc_covmat == 0.0].set(min_value)
    if log:
        plt.imshow(jnp.abs(loc_covmat), cmap='jet', norm='log')
    else:
        plt.imshow(loc_covmat, cmap='seismic', vmin=-max_abs, vmax=max_abs)
    plt.colorbar()
    plt.savefig(file_name)
    plt.close()
    gc.collect()

def variance_plots(variances, title, y_label, file_name):
    plt.figure()
    plt.grid()
    plt.title(title)
    components = np.arange(len(variances))+1
    plt.plot(components, variances)
    plt.xlabel('PCA component')
    plt.ylabel(y_label)
    plt.yscale('log')
    plt.savefig(file_name)
    plt.close()

    gc.collect()

def eigenvector_plots(eigenvectors, title, file_name):
    plt.figure()
    plt.grid()
    plt.title(title)
    x = np.arange(len(eigenvectors[0]))+1
    for i in range(len(eigenvectors)):
        plt.plot(x, eigenvectors[i], label='Component '+str(i))
    plt.legend()
    plt.xlabel('data bin')
    plt.ylabel('Eigenvector')
    plt.savefig(file_name)
    plt.close()

    gc.collect()


def plot_loglikes(loglikes, parameters, xlabel, file_name):
    plt.figure()
    plt.grid()
    plt.scatter(parameters, loglikes)
    plt.xlabel(xlabel)
    plt.ylabel('Loglike')
    plt.savefig(file_name)
    plt.close()

    gc.collect()

E_kin_energygrid = np.array([1.000000e+00, 1.400000e+00, 1.960000e+00, 2.744000e+00,
       3.841600e+00, 5.378240e+00, 7.529536e+00, 1.054135e+01,
       1.475789e+01, 2.066105e+01, 2.892547e+01, 4.049565e+01,
       5.669391e+01, 7.937148e+01, 1.111201e+02, 1.555681e+02,
       2.177953e+02, 3.049135e+02, 4.268789e+02, 5.976304e+02,
       8.366826e+02, 1.171356e+03, 1.639898e+03, 2.295857e+03,
       3.214200e+03, 4.499880e+03, 6.299831e+03, 8.819764e+03,
       1.234767e+04, 1.728674e+04, 2.420143e+04, 3.388201e+04,
       4.743481e+04, 6.640873e+04, 9.297222e+04, 1.301611e+05,
       1.822256e+05, 2.551158e+05, 3.571621e+05, 5.000269e+05,
       7.000377e+05, 9.800528e+05, 1.372074e+06, 1.920903e+06,
       2.689265e+06, 3.764971e+06, 5.270959e+06, 7.379343e+06,
       1.033108e+07, 1.446351e+07, 2.024892e+07, 2.834848e+07,
       3.968788e+07, 5.556303e+07, 7.778824e+07, 1.089035e+08,
       1.524649e+08, 2.134509e+08, 2.988313e+08, 4.183638e+08,
       5.857093e+08, 8.199931e+08, 1.147990e+09])


def plot_normalization(X, data, label):

    norm_factor = jnp.ones(len(data[0]))

    if label in ['tt', 'te', 'ee', 'bb', 'tb', 'eb']:
        norm_factor = X*(X+1)
    if label in ['pp','tp']:
        norm_factor = X*X*(X+1)*(X+1)
    if label in ['proton_unmodulated']:
        norm_factor = 1 # X*X    

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
    plt.plot(x,y)
    plt.fill_between(x, y - 3.*std, y + 3.*std, alpha = 0.3)
    plt.fill_between(x, y - std, y + std, alpha = 0.3)
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
        plt.xlabel('Data index')
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
            plt.plot(X, data[i]*norm_factor)
        plt.xlabel('Data index')
        plt.ylabel(label)
        plt.savefig(file_name)
        plt.close()

    gc.collect()

def data_plot_normalized(data, label, file_name):
    print(len(data.shape))
    print(len(data[0].shape))
    if data.shape[1] == 1:
        # just scatter the data
        plt.figure()
        plt.grid()
        plt.scatter(range(len(data)), data)
        plt.xlabel('Data index')
        plt.ylabel('Normalized '+label)
        plt.savefig(file_name)
        plt.close()

    else:
        N = len(data)
        X = np.array(range(len(data[0])))
        plt.figure()
        plt.grid()
        for i in range(N):
            plt.plot(X, data[i])
        plt.xlabel('Data index')
        plt.ylabel(label)
        plt.savefig(file_name)
        plt.close()

    gc.collect()


def pca_parameter_plot(x,y, x_label, y_label, title,file_name):
    plt.figure()
    plt.grid()
    plt.title(title)
    plt.scatter(x,y)
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
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig(file_name)
    plt.close()

    gc.collect()


def plot_pca_components_test_set(true, pred, pred_std, title, file_name):
    # plot residuals  
    plt.figure()
    plt.grid()
    plt.title(title)
    plt.errorbar(true, true-pred, yerr=pred_std, fmt='o')
    plt.xlabel('True')
    plt.ylabel('Residuals')
    plt.savefig(file_name)
    plt.close()

    gc.collect()

def plot_prediction_test(prediction, true, std, title, data_point, file_name):

    X = np.array(range(len(true[0])))
    norm_factor = plot_normalization(X, true, title)

    # make 2 subplots
    fig, ax = plt.subplots(nrows=3 , sharex=True, figsize=(10, 20))
    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    ax[0].set_title(title + ' \n ')

    # add text with str(data_point) into upper left corner
    ax[0].text(0.05, 0.95, 'Data point: ' + str(data_point), ha='left', va='bottom', transform=plt.gca().transAxes)

    if len(prediction) == 1:
        ax[0].errorbar([0], prediction*norm_factor, yerr=std, fmt='o', label='Prediction')
        ax[0].errorbar([0], true*norm_factor, fmt='o', label='True')

        # make residuals
        ax[1].errorbar([0], true*norm_factor-prediction*norm_factor, yerr=std*norm_factor, fmt='o')

        # make residuals 
        ax[2].errorbar([0], (true-prediction)/std, yerr=std/std, fmt='o')

    else:

        ax[0].plot(range(len(true[0])), prediction*norm_factor, label='Prediction')
        # make errorband around the prediction
        ax[0].fill_between(range(len(true[0])), (prediction-std)*norm_factor, (prediction+std)*norm_factor, alpha=0.5, label='1$\sigma$')
        ax[0].plot(range(len(true[0])), true[0]*norm_factor, label='True')

        # make residuals
        ax[1].plot(range(len(true[0])), true[0]*norm_factor-prediction*norm_factor, label='Residuals')
        ax[1].fill_between(range(len(true[0])), -std*norm_factor, std*norm_factor, alpha=0.5, label='1$\sigma$')

        # make residuals
        ax[2].plot(range(len(true[0])), (true[0]-prediction)/std, label='Residuals')
        ax[2].fill_between(range(len(true[0])), -1, 1, alpha=0.5, label='1$\sigma$')


    ax[2].set_xlabel('Data index')
    ax[0].set_ylabel('Prediction')
    ax[1].set_ylabel('Prediction - True')
    ax[2].set_ylabel('(Prediction - True) / $\sigma$')
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    plt.savefig(file_name)
    plt.close()

    gc.collect()