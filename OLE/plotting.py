import matplotlib.pyplot as plt
import numpy as np

plot_format = 'png'

def plot_normalization(X, data, label):

    if label in ['tt', 'te', 'ee', 'bb', 'tb', 'eb']:
        for i in range(len(data)):
            data[i] = data[i] * X*(X+1)
    if label in ['pp','tp']:
        for i in range(len(data)):
            data[i] = data[i] * X*X*(X+1)*(X+1)
    

    return data


def plain_plot(x, y, label, file_name):
    plt.figure()
    plt.plot(x, y, label=label)
    plt.legend()
    plt.savefig(file_name)
    plt.close()


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

        data = plot_normalization(X, data, label)

        plt.figure()
        plt.grid()
        for i in range(N):
            plt.plot(X, data[i])
        plt.xlabel('Data index')
        plt.ylabel(label)
        plt.savefig(file_name)
        plt.close()

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


def pca_parameter_plot(x,y, x_label, y_label, title,file_name):
    plt.figure()
    plt.grid()
    plt.title(title)
    plt.scatter(x,y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(file_name)
    plt.close()

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

def plot_pca_components_test_set(true, pred, pred_std, title, file_name):
    # plot residuals  
    plt.figure()
    plt.grid()
    plt.title(title)
    plt.errorbar(true, true-pred, yerr=pred_std, fmt='o')
    plt.xlabel('True')
    plt.ylabel('Residuals')
    plt.savefig(file_name)
