
# import the required libraries

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="darkgrid")

from statsmodels.tsa.seasonal import seasonal_decompose

class Seasonality:

    def seasonality(self, df_comp):
        output = "./Output/"
        # Naive decomposition Additive
        # observed = Trend + Sesonal + Residual
        additive = seasonal_decompose(df_comp.Healthcare, model="additive")
        additive.plot()
        plt.savefig(output+"seasonal_additive.png")
        # Naive decomposition Multiplicative
        # observed = Trend * Sesonal * Residual
        additive = seasonal_decompose(df_comp.Healthcare, model="multiplicative")
        additive.plot()
        plt.savefig(output+"seasonal_multiplicative.png")
