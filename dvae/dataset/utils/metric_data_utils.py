import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
import os

svc_datadir = './data/TeaMe_dataset_d15/teastore-webui' 

def plot_measurements_from_batch(batch, save_file_path):
     
    # generate some data for the curves
    x = np.linspace(0, 10, 100)
    n = 3  # number of curves to plot
    y = [np.sin(x), np.cos(x), np.tan(x)]  # list of y arrays

    # create a new figure and set its size
    fig = plt.figure(figsize=(8, 6))

    # plot the curves
    for i in range(n):
        plt.plot(x, y[i], label='curve {}'.format(i+1))

    # add a legend to the plot
    plt.legend()

    # add labels to the axes
    plt.xlabel('x')
    plt.ylabel('y')

    # save the plot to a file
    plt.savefig(save_file_path, dpi=300, bbox_inches='tight')




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
    #scaler = preprocessing.MaxAbsScaler()
    #y_scaled = scaler.fit_transform(np.array(y).reshape(-1, 1))

    fig, ax = plt.subplots()
    ax.plot(x, y)

    ax.set(xlabel='timestamp', ylabel='value',
           title=metric_name)
    ax.grid()

    fig.savefig(save_file_path)
    plt.close()


if __name__ == '__main__':
    dirs = os.listdir(svc_datadir)
    files = []
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