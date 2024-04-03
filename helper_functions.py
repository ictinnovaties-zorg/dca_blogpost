# Some helpder functions to calculate some of the metrics used in the blogpost

from sklearn.metrics import recall_score, f1_score, roc_auc_score
from itertools import product
import numpy as np
import pandas as pd

def net_benefit(sensitivity, specificity, prevalence, threshold):
    return (sensitivity * prevalence) - ((1 - specificity) * (1 - prevalence)) * (threshold / (1 - threshold))

# helper function to get sensitivity and specificity for a given prediction
def get_sensitivity_specificity(pred, true):
    sensitivity = recall_score(true, pred)
    specificity = recall_score(np.logical_not(true), np.logical_not(pred))
    return sensitivity, specificity

# Helper function to calculate net benefit for a given input vector of prediction probabilities
def calc_net_benefit(input_prob, true, thold):
    prevalence = true.sum() / len(true)
    sensitivity, specificity = zip(*[get_sensitivity_specificity((input_prob >= prob).astype(int), true) for prob in thold])
    return net_benefit(np.array(sensitivity), np.array(specificity), prevalence, thold)

# Calculate accuracy for a given input vector of prediction probabilities
def calc_accuracy(input_prob, true, thold):
    return [((input_prob >= prob).astype(int) == true).sum() / len(input_prob) for prob in thold]

# Calculate f1 score for a given input vector of prediction probabilities
def calc_f1(input_prob, true, thold):
    return [f1_score(true, (input_prob >= prob).astype(int), pos_label=1) for prob in thold]

# Note we ignore the thold values here, roc_auc already iterates over the thresholds
def calc_auc(input_prob, true, thold):
    return [roc_auc_score(true, input_prob)]

# Calculate the metrics for a given set of input probablities
def calc_metrics(input_prob, true, metrics, thold=np.array([0.5]), e=1e-6, add_everyone_noone=True):
    # Some of the metrics are not stable for 0 or 1 threshold
    thold[thold == 0] = e
    thold[thold == 1] = 1 - e

    # The rest of the code assumes metrics is a list,so if it is
    # a string, convert it to a list of length 1
    if isinstance(metrics, str):
        metrics = [metrics]
    # Link the metric names to the functions that calculate them, only keep the ones the user actually wants
    # TODO: add assertion that all metrics are in the possible metrics dict
    possible_metrics = {'net_benefit': calc_net_benefit, 
                        'accuracy': calc_accuracy, 'f1': calc_f1,
                        'auc': calc_auc}
    metrics_dict = dict(zip(metrics, [possible_metrics[m] for m in metrics]))

    if add_everyone_noone:
        input_prob['everyone'] = np.ones(len(true))
        input_prob['noone'] = np.zeros(len(true))

    metric_data = pd.DataFrame([m(ip, true, thold) for ip, m in product(input_prob.values(), metrics_dict.values())]).T
    metric_data.columns = [colname + '_' + postifx for postifx, colname in product(input_prob.keys(), metrics_dict.keys())]

    metric_data.index = thold
    return metric_data

def thold_to_odds(thold):
    return 1/(thold / (1-thold))

def dca(input_prob, true, thold_range=[0,0.35], n=100, thold_odds=False, **plotargs):
    thold = np.linspace(thold_range[0], thold_range[1], n)
    dca_plot_values = calc_metrics(input_prob, true, metrics='net_benefit', thold=thold)
    if thold_odds:
        # Convert the threshold to odds if requested
        # TODO: Does work, but plot is ugly because the x-axis is log and the axis flips. Fix later. 
        # dca_plot_values.index = thold_to_odds(dca_plot_values.index)
        pass
    return dca_plot_values.plot(**plotargs)

def prob_density_plot(df, y, by, title='Density plot', vline=[0.5, 0.2], labels=['No Cancer', 'Cancer'], bw_method=0.15):
    fig, ax = (df[[by, y]]
        .groupby(by)[y]
        .plot.kde(legend=True, xlim=(0,1), bw_method=bw_method, title=title))
    for v in vline:
        ax.axvline(x=v, color='black', linestyle='--')
    ax.legend(labels)
    return fig