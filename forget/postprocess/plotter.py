import numpy as np
from matplotlib.pyplot import figure
plt.rcParams['figure.figsize'] = [10, 7]

class processMeasurements:
    def __init__(self, forget_msrmt):
        self.forget_stats = forget_msrmt.forgetStatistics
        self.sum_over_ep_Forget = torch.sum(self.forget_stats, 0)
    
    def plotForgetHist(self):
        length = len(torch.flatten(self.sum_over_ep_Forget))
        hist = plt.hist(torch.flatten(self.sum_over_ep_Forget), alpha=0.5, weights = np.ones(length)/length)
        plt.ylabel('Fraction of total events')
        plt.xlabel('Number of forgetting events')
        plt.show()