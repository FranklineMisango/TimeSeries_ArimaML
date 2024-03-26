

# import the required libraries

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="darkgrid")
import scipy.stats
import pylab

import statsmodels.tsa.stattools as sts

class Stationarity:

    def stationarity(self, df_comp):
        output = "./Output/"
        # Stationarity
        # AD fuller test for stationarity
        sts.adfuller(df_comp.wn)
        # AD fuller test for stationarity
        sts.adfuller(df_comp.Healthcare)
        # The QQ plot for gausian test
        scipy.stats.probplot(df_comp.wn, plot=pylab)
        plt.title("QQ plot for White Noise")
        pylab.savefig(output+"qq_wn.png")
        # The QQ plot
        scipy.stats.probplot(df_comp.Healthcare, plot=pylab)
        plt.title("QQ plot for Healthcare")
        pylab.savefig(output+"qq_healthcare.png")

