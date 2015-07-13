import problem_set_answers as a
import pprint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import json

import importlib
importlib.reload(a)

######################
### Wrangling data ###
######################

#-- Read data into pandas from CSV
# data = pd.read_csv(r'turnstile_data_master_with_weather.csv')
# data = pd.read_csv(r'turnstile_weather_v2.csv')

#-- Modity data to add a day of the week, then save to new CSV
# data['day_of_week'] = pd.to_datetime(data['DATEn']).dt.dayofweek
# data.to_csv(path_or_buf=r'turnstile_data_working_copy.csv')
#-- Use the data with day of week already there
# data = pd.read_csv(r'turnstile_data_working_copy.csv')
# del data['Unnamed: 0']

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

# features_to_explor = ['UNIT', 'day_of_week', 'Hour', 'maxpressurei', 'maxdewpti', 'mindewpti', 'minpressurei', 'meandewpti', 'meanpressurei', 'fog', 'rain', 'meanwindspdi', 'mintempi', 'meantempi', 'maxtempi', 'precipi']


### Plot the interesting factors against ENTRIESn_hourly
# Figures generated are in dir EDA_figs

# for feature in features_to_explore:
# 	a.bar_plot_mean_Entries(data, feature)


### Looking at only the binary factors (Rain and Fog)
#-- Select out the two sections for each one:
# no_rain = data[data.rain = ]['ENTRIESn_hourly']
# rain = data[data.rain = ]['ENTRIESn_hourly']
# no_fog = data[data.fog = ]['ENTRIESn_hourly']
# fog = data[data.fog = ]['ENTRIESn_hourly']

#-- Test selection: make a hist of the data
# a.hist_MWW_suitability(no_rain, rain, rORf='fog')
# a.hist_MWW_suitability(no_fog, fog, rORf='fog')

#-- compare binary factors using Mann-Whitney statistic
# bi_f = {'title': ['mean with_rain',
# 				  'mean without_rain',
# 				  'Mann-Whitney U-statistic',
# 				  'Mann-Whitney p-value' ]}
# bi_f['rain'] = list(a.mann_whitney_plus_means(no_rain, rain))
# bi_f['fog'] = list(a.mann_whitney_plus_means(no_fog, fog))
# pprint.pprint (bi_f)
	# >{'title': ['mean with_rain',
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

###

#########################
### Linear Regression ###
#########################

### Prepare data for linear regression ###
#-- Create dummy varriables for 'UNIT'
# data, UNIT_dummy = a.UNIT_dummy_vars(data)

## save the dummy variables used for reference
# with open('UNIT_dummy.json', 'wr') as f:
#     json.dump(UNIT_dummy, f)

## save over working file
# data['day_of_week'] = pd.to_datetime(data['DATEn']).dt.dayofweek
# data.to_csv(path_or_buf=r'turnstile_data_working_copy.csv')

## Add UNIT_dummy and day_of_week columns for test_data
# test_data = pd.read_csv(r'turnstile_weather_v2.csv')
# with open('UNIT_dummy.json') as f:
#     UNIT_dummy = json.load(f)
# test_data, UNIT_dummy = a.UNIT_dummy_vars(test_data, UNIT_dummy)
# test_data['day_of_week'] = pd.to_datetime(test_data['DATEn']).dt.dayofweek
# change hour to Hour so data & test_data match
# test_data = test_data.rename(columns={'hour': 'Hour'})
# test_data = test_data.rename(columns={'meanwspdi': 'meanwindspdi'})
# test_data.to_csv(path_or_buf = r'turnstile_weather_v2_working_copy.csv')


### Find the right features to use

#-- List of features to explore
## The following columns were removed due to lack of test data: 'maxpressurei', 'maxdewpti', 'mintempi',  'mindewpti', 'minpressurei', 'meandewpti', 'maxtempi'
# features_to_explore = ['UNIT_dummy', 'day_of_week', 'Hour', 'meanpressurei', 'fog', 'rain', 'meanwindspdi', 'meantempi', 'precipi']

#-- save reslts in a list in the form (r_squared, [feature list], (intercept, params))
# results = [('r_squared', ('feature','list'), ('intercept', 'params')),]

## reload the data
# data = pd.read_csv(r'turnstile_data_working_copy.csv')
# del data['Unnamed: 0']
# test_data = pd.read_csv(r'turnstile_weather_v2_working_copy.csv')
# del test_data['Unnamed: 0']

## Create numpy arrays
# values_array = data['ENTRIESn_hourly'].values
# test_values_array = test_data['ENTRIESn_hourly'].values

### Test every variable independently against the

# for feature in features_to_explore:
#     #-- extract feature
#     #-- generate predictions
#     feature_array = data[feature].values
#     intercept, params = a.OLS_linear_regression(feature_array, values_array)
#
#     #-- calculate r** using backup data
#     test_feature_array = test_data[feature].values
#     predictions = test_feature_array * params + intercept
#     r_squared = a.compute_r_squared(test_values_array, predictions)
#     #-- append results to list
#     results.append((r_squared, ([feature],), (intercept, tuple(params.tolist()))))

    # [('r_squared', ('feature', 'list'), ('intercept', 'params')),
    #  (0.36066468329622159, (['UNIT_dummy'],), (0.44582044219855199, (1.000019654833768,))),
    #  (-0.062813493262236841, (['day_of_week'],), (1350.5997806795308, (-85.5451482036722,))),
    #  (-0.020645166615559596, (['Hour'],), (447.17776398603291, (59.48616831074126,))),
    #  (-0.072148096599663258, (['meanpressurei'],), (9752.6656708963455, (-288.91356369944765,))),
    #  (-0.073920312745420658, (['fog'],), (1083.449282087679, (71.21006754320405,))),
    #  (-0.072021772467172562, (['rain'],), (1090.2787801517247, (15.167596594046824,))),
    #  (-0.067979992755955676, (['meanwindspdi'],), (921.35747190610505, (31.388951556096405,))),
    #  (-0.069465093618023444, (['meantempi'],), (1616.6334487585414, (-8.11089419107536,))),
    #  (-0.073463327604493589, (['precipi'],), (1086.2781786381292, (52.64972158324689,)))]


################################
### Linear Regression Take 2 ###
################################

### Well that didn't work :( (lots of -ve R values)
### going to try again with more dummy variables, SGD & spliting the data into test data and learning data instead of using the second data set.
# http://scikit-learn.org/stable/tutorial/machine_learning_map/index.html

### Prepare data for linear regression ###
## add more dummy units for 'UNIT', 'day_of_week', 'Hour' using mean ENTRIESn_hourly
# use original data
# data = pd.read_csv(r'turnstile_data_master_with_weather.csv')
# data['day_of_week'] = pd.to_datetime(data['DATEn']).dt.dayofweek
# del data['Unnamed: 0']

# mean dummy set for UNIT, day_of_week, & Hour
# data, UNIT_means = a.mean_dummy_units(data, feature='UNIT')
# data, day_of_week_means = a.mean_dummy_units(data, feature='day_of_week')
# data, Hour_means = a.mean_dummy_units(data, feature='Hour')

## save data to working data
# data.to_csv(path_or_buf=r'turnstile_data_working_copy.csv')

## save the dummy variables used for reference
mean_dummys = {'UNIT_means': a.JSONify_dict(UNIT_means),
               'day_of_week_means': a.JSONify_dict(day_of_week_means),
               'Hour_means': a.JSONify_dict(Hour_means)}

with open('mean_dummys.json', 'w') as f:
    json.dump(mean_dummys, f)

## reload the data without test data
data = pd.read_csv(r'turnstile_data_working_copy.csv')
del data['Unnamed: 0']

## split data into training_data and test_data
# http://stackoverflow.com/questions/12190874/pandas-sampling-a-dataframe
# test_rows = np.random.choice(data.index.values, len(data)*0.1, replace=False)
# with open('test_rows.json', 'w') as f:
#     json.dump(test_rows.tolist(), f)

# Split data from test_rows
with open('test_rows.json') as f:
    test_rows = json.load(f)
test_data = data.ix[test_rows]
training_data = data.drop(test_rows)
