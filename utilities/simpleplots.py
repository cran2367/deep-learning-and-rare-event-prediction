import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

##############################
##### Plotting functions #####
##############################


def plot_metric(model_history, metric, ylim=None, grid=False):
    sns.set()
    
    if grid is False:
        sns.set_style("white")
        sns.set_style("ticks")

    train_values = [
        value for key, value in model_history.items() if metric in key.lower()
    ][0]
    valid_values = [
        value for key, value in model_history.items() if metric in key.lower()
    ][1]

    fig, ax = plt.subplots()

    color = 'tab:blue'
    ax.set_xlabel('Epoch', fontsize=16)
    ax.set_ylabel(metric, color=color, fontsize=16)

    ax.plot(train_values, '--', color=color, label='Train ' + metric)
    ax.plot(valid_values, color=color, label='Valid ' + metric)
    ax.tick_params(axis='y', labelcolor=color)
    ax.tick_params(axis='both', which='major', labelsize=14)

    if ylim is None:
        ylim = [
            min(min(train_values), min(valid_values), 0.),
            max(max(train_values), max(valid_values))
        ]
    plt.yticks(np.round(np.linspace(ylim[0], ylim[1], 6), 1))
    plt.legend(loc='upper left', fontsize=16)
    
    if grid is False:
        sns.despine(offset=1, trim=True)

    return plt, fig


def plot_model_recall_fpr(model_history, grid=False):
    sns.set()
    
    if grid is False:
        sns.set_style("white")
        sns.set_style("ticks")
    
    train_recall = [
        value for key, value in model_history.items()
        if 'recall' in key.lower()
    ][0]
    valid_recall = [
        value for key, value in model_history.items()
        if 'recall' in key.lower()
    ][1]

    train_fpr = [
        value for key, value in model_history.items()
        if 'false_positive_rate' in key.lower()
    ][0]
    valid_fpr = [
        value for key, value in model_history.items()
        if 'false_positive_rate' in key.lower()
    ][1]

    fig, ax = plt.subplots()

    color = 'tab:red'
    ax.set_xlabel('Epoch', fontsize=16)
    ax.set_ylabel('value', fontsize=16)
    ax.plot(train_recall, '--', color=color, label='Train Recall')
    ax.plot(valid_recall, color=color, label='Valid Recall')
    ax.tick_params(axis='y', labelcolor='black')
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(loc='upper left', fontsize=16)

    color = 'tab:blue'
    ax.plot(train_fpr, '--', color=color, label='Train FPR')
    ax.plot(valid_fpr, color=color, label='Valid FPR')
    plt.yticks(np.round(np.linspace(0., 1., 6), 1))

    fig.tight_layout()
    plt.legend(loc='upper left', fontsize=16)
    
    if grid is False:
        sns.despine(offset=1, trim=True)
    
    return plt, fig