import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
import os
import math

root_dir = '/home/ggrabher/Code/the-prometheus-metrics-dataset/'
svc_datadir = '5-minutes-metrics/teastore/teastore-webui' 

def plot_n_grids_two_curves_subgraph(y_orig, y_prime, rows, cols, y_names, savepath):
    
    fig, ax = plt.subplots(rows, cols, sharex='col', sharey='row', figsize=(20,10))

    seq_len = y_orig.shape[1]
    x_arr = range(0, seq_len)

    y_count = 0
    for i in range(rows):
        for j in range(cols):
            ax[i, j].set(xlabel='time(s)', ylim=(0,1))
            ax[i, j].plot(x_arr, y_orig[y_count,:], label='Original')
            ax[i, j].plot(x_arr, y_prime[y_count,:], label='Generated')
            ax[i, j].legend(title = y_names[y_count])

            y_count = y_count + 1

    if savepath != '':
        plt.savefig(savepath, dpi=(200))

    plt.close()


def plot_n_batches_two_curves_subgraph(nb_batches, seq_len, line_len, y_orig, y_prime, title, savepath):
    cols = line_len
    rows = math.floor(nb_batches / line_len)

    fig, ax = plt.subplots(rows, cols, sharex='col', sharey='row', figsize=(20,10))

    #seq_len = y_orig.shape[1]
    x_arr = range(0, seq_len)

    y_start = 0
    y_end = seq_len
    for i in range(rows):
        for j in range(cols):
            ax[i, j].set(xlabel='time(s)', ylim=(0,1))
            ax[i, j].plot(x_arr, y_orig[y_start:y_end], label='Original')
            ax[i, j].plot(x_arr, y_prime[y_start:y_end], label='Generated')
            ax[i, j].legend(title = "Batch {}".format(i*line_len + j))

            y_start = y_end
            y_end = y_end + seq_len 

    if savepath != '':
        plt.savefig(savepath, dpi=(200))

    plt.close()


def plot_two_curves_graph(x_arr, y1_arr, y2_arr, x_label='x', y_label='y', y1_label='y1', y2_label='y2', title='x y1 y2 Graph', savepath='' ):

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set(xlabel=x_label, ylabel=y_label, ylim=(0,1))
    ax.plot(x_arr, y1_arr, label=y1_label)
    ax.plot(x_arr, y2_arr, label=y2_label)
    ax.legend(title=title, loc="upper left")
    if savepath != '':
        plt.savefig(savepath)
    plt.close()


def plot_measurements_from_json(file_path, save_file_path):
    """
    Reads a JSON file containing a list of measurements and plots it to a PNG file using Matplotlib.

    Args:
        file_path (str): The file path to read the JSON file from.
        save_file_path (str): The file path to save the PNG image file to.
    """
    metric_name = 'measures'
    with open(file_path, 'r') as f:
        data = json.load(f)
        results = data['data']['result']
        metric_name = os.path.basename(file_path).split('.')[0]

    x = range(len(results[0]['values']))
    y = [r[1] for r in results[0]['values']]
    y = np.array(y, dtype=np.float32)
    scaler = preprocessing.MaxAbsScaler()
    y_scaled = scaler.fit_transform(np.array(y).reshape(-1, 1))

    fig, ax = plt.subplots()
    ax.set(ylim=(0,1))
    ax.plot(x, y_scaled)

    ax.set(xlabel='timestamp', ylabel='value',
           title=metric_name)
    ax.grid()

    fig.savefig(save_file_path)
    plt.close()


if __name__ == '__main__':
    full_dir_path = root_dir + svc_datadir
    dirs = os.listdir(full_dir_path)
    files = []
    """
    for dir in dirs:
        f_in_dir = os.listdir(svc_datadir+'/'+dir)
        files.extend([dir+'/'+f for f in f_in_dir if os.path.isfile(svc_datadir+'/'+dir+'/'+f)])

    # For metric in metric_names:
    for exp_file in files:
        # plot the measurements from the JSON file and save as a PNG file
        json_file_path = svc_datadir + '/' + exp_file
        exp_name, f_name = exp_file.split('/')
        if not(os.path.isdir('./data/tmp/' + exp_name)):
                os.makedirs('./data/tmp/' + exp_name)
        save_file_path = './data/tmp/' + exp_name + '/' + f_name.split('.')[0] + '.png' 
        plot_measurements_from_json(json_file_path, save_file_path)
    """
    for dirpath, dirnames, filenames in os.walk(full_dir_path):
        for filename in filenames:
            if filename.endswith('.json'):
                filepath = os.path.join(dirpath, filename)
                files.append(filepath) 

    for fpath in files:
        # plot the measurements from the JSON file and save as a PNG file
        dirpath = os.path.dirname(fpath)
        dirpath = dirpath[len(root_dir):]
        if not(os.path.isdir('./data/tmp/dataset_plots/' + dirpath)):
                os.makedirs('./data/tmp/dataset_plots/' + dirpath)

        fname = os.path.basename(fpath).split('.')[0]
        save_file_path = './data/tmp/dataset_plots/' + dirpath + '/' + fname + '.png' 
        plot_measurements_from_json(fpath, save_file_path)