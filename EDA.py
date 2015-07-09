import problem_set_answers as a
import pprint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
### Analyze the binary features ['rain', 'fog', 'thunder']

######################
### Wrangling data ###
######################

#Read data into pandas from CSV
#data = pd.read_csv(r'turnstile_data_master_with_weather.csv')
# data = pd.read_csv(r'turnstile_weather_v2.csv')

# Modity data to add a day of the week, then save to new CSV
#data['day_of_week'] = pd.to_datetime(data['DATEn']).dt.dayofweek
#data.to_csv(path_or_buf=r'turnstile_data_master_with_weather_dayofweek.csv')
# Use the data with day of week already there
data = pd.read_csv(r'turnstile_data_master_with_weather_dayofweek.csv')

######################
### Exploring data ###
######################

# ### What Features might be affecting ridership?

# all possible Features:
# print (data.columns.values.tolist())
#> ['Unnamed: 0', 'Unnamed: 0.1', 'UNIT', 'DATEn', 'TIMEn', 'Hour', 'DESCn', 'ENTRIESn_hourly', 'EXITSn_hourly', 'maxpressurei', 'maxdewpti', 'mindewpti', 'minpressurei', 'meandewpti', 'meanpressurei', 'fog', 'rain', 'meanwindspdi', 'mintempi', 'meantempi', 'maxtempi', 'precipi', 'thunder', 'day_of_week']

# Which ones should we explore?
# UNIT: YES. Using UNIT would tell us if their is a specific area that is being used more than another not weather related we want to predict ridership overall.
# DATEn: NO. Would tell us if specific days are used more than others. This would be useful if we could seperate days of the week.(Done) Use day_of_week instead.
# day_of_week: YES. this is DATEn in usable format
# TIMEn: NO. Would be useful if it was grouped by hours. Use Hour instead.
# Hour: YES. this is TIMEn in usable format
# YES to all weather data except thunder:
#     'maxpressurei', 'maxdewpti', 'mindewpti', 'minpressurei', 'meandewpti', 'meanpressurei', 'fog', 'rain', 'meanwindspdi', 'mintempi', 'meantempi', 'maxtempi', 'precipi'

# DESCn: NO. It is only ever REGULAR
# ENTRIESn_hourly: NO. primary dependent variable
# EXITSn_hourly: NO. secondary dependent variable
# thunder: NO. Only ever 0

features_to_explore = ['UNIT', 'day_of_week', 'Hour', 'maxpressurei', 'maxdewpti', 'mindewpti', 'minpressurei', 'meandewpti', 'meanpressurei', 'fog', 'rain', 'meanwindspdi', 'mintempi', 'meantempi', 'maxtempi', 'precipi']


### Plot the interesting factors against ENTRIESn_hourly
# Figures generated are in dir EDA_figs

# for feature in features_to_explore:
# 	a.bar_plot_mean_Entries(data, feature)


### compare binary factors using Mann-Whitney statistic
# bi_f = {'title': ['mean with_rain',
# 				  'mean without_rain',
# 				  'Mann-Whitney U-statistic',
# 				  'Mann-Whitney p-value' ]}
# bi_f['rain'] = list(a.mann_whitney_plus_means(data[data.rain == 0]['ENTRIESn_hourly'], data[data.rain == 1]['ENTRIESn_hourly']))
# bi_f['fog'] = list(a.mann_whitney_plus_means(data[data.fog == 0]['ENTRIESn_hourly'], data[data.fog == 1]['ENTRIESn_hourly']))
# pprint.pprint (bi_f)
### result from 'turnstile_data_master_with_weather.csv'
# {'title': ['mean with_rain',
#            'mean without_rain',
#            'Mann-Whitney U-statistic',
#            'Mann-Whitney p-value'],
#  'fog': [1083.4492820876781,
#          1154.6593496303688,
#          1189034717.5,
#          6.0915569104373036e-06],
#  'rain': [1090.278780151855,
#           1105.4463767458733,
#           1924409167.0,
#           0.024999912793489721]}
