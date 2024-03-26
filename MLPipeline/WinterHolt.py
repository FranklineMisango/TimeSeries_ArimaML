
# import the required libraries
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
sns.set_theme(style="darkgrid")



class Winterholt:

    def holt(self, df_comp):
        hw_model = ExponentialSmoothing(df_comp.Healthcare.tolist())
        model_fit = hw_model.fit()
        # make prediction
        yhat = model_fit.predict(1, len(df_comp))
        # we are calling the model to predict all datapoints that are same as the dataset to see the model's performance
        plt.figure(figsize=(20, 5))
        plt.plot(df_comp.Healthcare.tolist())
        plt.plot(yhat.tolist(), color='red')
        plt.title("Holt Winter Model Prediction Vs Actual Healthcare")
        plt.legend(["actual", "predicted"])
        plt.savefig("Output/holtwinter.png")

