import  pandas as pd
import matplotlib.pyplot as plt
from MLPipeline.Stationarity import Stationarity
from MLPipeline.RandomWalk import RandomWalk
from MLPipeline.WhiteNoise import WhiteNoise
from MLPipeline.Seasonality import Seasonality
from MLPipeline.WinterHolt import Winterholt
from MLPipeline.ARIMA import ARIMA_Model

# importing the data
raw_csv_data = pd.read_excel("./Input/CallCenterData.xlsx")

# check point of data
df_comp = raw_csv_data.copy()

df_comp.set_index("month", inplace=True)

# seeting the frequency as monthly
df_comp = df_comp.asfreq('M')


df_comp.Healthcare.plot(figsize=(20,5), title="Healthcare")
plt.savefig("Output/healthcare.png")


WhiteNoise().white_noise(df_comp)

RandomWalk().random_walk()

Stationarity().stationarity(df_comp)

Seasonality().seasonality(df_comp)

Winterholt().holt(df_comp)

ARIMA_Model().compute(df_comp)