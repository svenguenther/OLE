import matplotlib.pyplot as plt
import numpy as np
import gc
import jax.numpy as jnp

plot_format = 'png'


def plot_loglikes(loglikes, parameters, xlabel, file_name):
    plt.figure()
    plt.grid()
    plt.scatter(parameters, loglikes)
    plt.xlabel(xlabel)
    plt.ylabel('Loglike')
    plt.savefig(file_name)
    plt.close()

    gc.collect()


def plot_normalization(X, data, label):

    norm_factor = jnp.ones(len(data[0]))

    if label in ['tt', 'te', 'ee', 'bb', 'tb', 'eb']:
        norm_factor = X*(X+1)
    if label in ['pp','tp']:
        norm_factor = X*X*(X+1)*(X+1)
    if label in ['proton_unmodulated']:
        norm_factor = X*X    

    return norm_factor


def plain_plot(x, y, label, file_name):
    plt.figure()
    plt.plot(x, y, label=label)
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