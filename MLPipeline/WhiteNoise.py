
# import the required libraries
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
sns.set_theme(style="darkgrid")
import statsmodels.graphics.tsaplots as sgt

class WhiteNoise:



    # generating a white noise data for the Healthcare attribute

    def white_noise(self, df_comp):
        output = "./Output/"
        wn = np.random.normal(loc=df_comp.Healthcare.mean(), scale=df_comp.Healthcare.std(), size=len(df_comp))
        df_comp["wn"] = wn
        df_comp.wn.plot(figsize=(20, 5))
        plt.title("White noise time-series", size=24)
        plt.savefig(output+"whitenoise.png")
        autocorrelation_plot(df_comp.wn)
        plt.savefig(output+"autocorr_wn.png")
        autocorrelation_plot(df_comp.Healthcare)
        plt.savefig(output+"autocorr_health.png")
        sgt.plot_acf(df_comp.wn, zero=False, lags=40)
        plt.title("ACF Of WN", size=20)
        plt.savefig(output+"act_wn.png")
        sgt.plot_acf(df_comp.Healthcare, zero=False, lags=40)
        plt.title("ACF Of Healthcare", size=20)
        plt.savefig(output+"acf_health.png")

