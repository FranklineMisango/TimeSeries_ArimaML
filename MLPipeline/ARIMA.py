# import the required libraries

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="darkgrid")

import statsmodels.graphics.tsaplots as sgt
from  statsmodels.tsa.arima.model import ARIMA
from scipy.stats.distributions import chi2
from statsmodels.tsa.statespace.sarimax import SARIMAX

class ARIMA_Model:

    # ARIMA

    def LLR_test(self, mod_1, mod_2, DF=1):
        L1 = mod_1.fit().llf
        L2 = mod_2.fit().llf
        LR = (2 * (L2 - L1))
        p = chi2.sf(LR, DF).round(3)
        return p

    def compute(self, df_comp):
        output = "./Output/"

        model_ar_1_i_1_ma_1 = ARIMA(df_comp.Healthcare, order=(1, 1, 1))
        results_ar_1_i_1_ma_1 = model_ar_1_i_1_ma_1.fit()
        results_ar_1_i_1_ma_1.summary()

        model_ar_1_i_1_ma_2 = ARIMA(df_comp.Healthcare, order=(1, 1, 2))
        results_ar_1_i_1_ma_2 = model_ar_1_i_1_ma_2.fit()

        model_ar_2_i_1_ma_1 = ARIMA(df_comp.Healthcare, order=(2, 1, 1))
        results_ar_2_i_1_ma_1 = model_ar_2_i_1_ma_1.fit()

        model_ar_2_i_1_ma_2 = ARIMA(df_comp.Healthcare, order=(2, 1, 2))
        results_ar_2_i_1_ma_2 = model_ar_2_i_1_ma_2.fit()

        print("ARIMA(1,1,2):  \t LL = ", results_ar_1_i_1_ma_2.llf, "\t AIC = ", results_ar_1_i_1_ma_2.aic)
        print("ARIMA(2,1,1):  \t LL = ", results_ar_2_i_1_ma_1.llf, "\t AIC = ", results_ar_2_i_1_ma_1.aic)
        print("ARIMA(2,1,2):  \t LL = ", results_ar_2_i_1_ma_2.llf, "\t AIC = ", results_ar_2_i_1_ma_2.aic)

        # Check with LLR test
        print("\nLLR test p-value = " + str(self.LLR_test(model_ar_1_i_1_ma_1, model_ar_2_i_1_ma_2, DF=2)))

        # Check with LLR test
        print("\nLLR test p-value = " + str(self.LLR_test(model_ar_2_i_1_ma_1, model_ar_2_i_1_ma_2, DF=1)))

        # Residual ARIMA(2,1,1)
        df_comp['res_ar_2_i_1_ma_1'] = results_ar_2_i_1_ma_1.resid
        sgt.plot_acf(df_comp.res_ar_2_i_1_ma_1, zero=False, lags=40)
        plt.title("ACF Of Residuals for ARIMA(2,1,1)", size=20)
        plt.savefig(output+"acf_211.png")

        # higher level

        model_ar_1_i_2_ma_1 = ARIMA(df_comp.Healthcare, order=(1, 2, 1))
        results_ar_1_i_2_ma_1 = model_ar_1_i_2_ma_1.fit()
        results_ar_1_i_2_ma_1.summary()

        df_comp['res_ar_1_i_2_ma_1'] = results_ar_1_i_2_ma_1.resid.iloc[:]
        sgt.plot_acf(df_comp.res_ar_1_i_2_ma_1[2:], zero=False, lags=40)
        plt.title("ACF Of Residuals for ARIMA(1,2,1)", size=20)
        plt.savefig(output+"acf_121.png")

        # ARIMAX

        self.arimax(df_comp)

        self.sARIMAX(df_comp)

    def sARIMAX(self, df_comp):
        model_sarimax = SARIMAX(df_comp.Healthcare, exog=df_comp.Banking, order=(1, 1, 1), seasonal_order=(2, 0, 1, 5))
        results_sarimax = model_sarimax.fit()
        results_sarimax.summary()

    def arimax(self, df_comp):
        output = "./Output/"
        model_ar_1_i_1_ma_1_X = ARIMA(df_comp.Healthcare, exog=df_comp.Banking, order=(1, 1, 1))
        results_ar_1_i_1_ma_1_X = model_ar_1_i_1_ma_1_X.fit()
        results_ar_1_i_1_ma_1_X.summary()
        df_comp['resX_ar_1_i_1_ma_1'] = results_ar_1_i_1_ma_1_X.resid.iloc[:]
        sgt.plot_acf(df_comp.resX_ar_1_i_1_ma_1, zero=False, lags=40)
        plt.title("ACF Of Residuals for ARIMAX(1,1,1)", size=20)
        plt.savefig(output+"acfx_111.png")


