import numpy as np
import pandas as pd

from utilities.datapreprocessing import flatten
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

from pylab import rcParams
rcParams['figure.figsize'] = 8, 6
plt.rcParams.update({'font.size': 22})

def reconstructionerror_vs_class(model, sample, y):
  reconstruction = model.predict(sample)
  
  if(len(sample.shape) == 3):
      sample = flatten(sample)
      reconstruction = flatten(reconstruction)
      
  reconstruction_error = np.mean(np.power(sample - reconstruction, 2), axis=1)

  return pd.DataFrame({'reconstruction_error': reconstruction_error,
                       'true_class': y})
  
  
def error_boxplot(error_vs_class, ylim=None):
  fig, ax = plt.subplots(1,1, sharey=False)

  bp = error_vs_class.boxplot(by="true_class",ax=ax,figsize=(6,8))
  ax.set_xlabel('true class')

  if ylim is not None:
    ax.set_ylim(ylim[0], ylim[1])
  
  plt.show()
  return plt, fig


# Ref:
#
# 1. Higgins, J. P., & Green, S. (Eds.). (2011). Cochrane handbook for systematic reviews of interventions (Vol. 4). John Wiley & Sons.
#
# 2. https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/1471-2288-14-135
#
# Shows the standard deviation can be estimated as (q3-q1)/1.35 [1,2]
#
# We will get the summary statistic for the two classes reconstruction errors.
#
# Class 0:
# median0 = quantile(0.5)
# sd0 = (q3-q1)/1.35
#
# Class 1:
# median1 = quantile(0.5)
# sd1 = (q3-q1)/1.35
#
# Difference between them:
# t-statistic = (median1 - median0) / sqrt( sd1^2 / n1 + sd0^2 / n2 )

def robust_statistics(z):
  median = z.quantile(0.5)
  iqr = z.quantile(0.75) - z.quantile(0.25)  # Interquartile range
  sd = iqr/1.35
  return {'median': median, 'sd': sd}
  
def class_errors(error_vs_class, label): 
  return error_vs_class.loc[error_vs_class['true_class'] == label, 'reconstruction_error']

def robust_stat_difference(z1, z2):
  stats_z1, stats_z2 = robust_statistics(z1), robust_statistics(z2)

  robust_t_statistic = (stats_z2['median'] - stats_z1['median']) / np.sqrt((stats_z2['sd']**2) / len(z2) + (stats_z1['sd']**2) / len(z1))

  return robust_t_statistic
  
def model_confusion_matrix(error_vs_class, threshold):
  pred_y = [1 if e > threshold else 0 for e in error_vs_class.reconstruction_error.values]
  
  conf_matrix = confusion_matrix(error_vs_class.true_class, pred_y)

  labels = ["Normal","Break"]
  
  fig, ax = plt.subplots(figsize=(4, 4))
  
  sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt="d");
  ax.set_ylabel('True class')
  ax.set_xlabel('Predicted class')
  fig.tight_layout()

  return conf_matrix, fig