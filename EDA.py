import problem_set_answers as a
import pandas as pd
import numpy as np
import json
# import matplotlib.pyplot as plt
# import matplotlib.mlab as mlab
# import pprint

# import importlib
# importlib.reload(a)

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
# mean_dummys = {'UNIT_means': a.JSONify_dict(UNIT_means),
#                'day_of_week_means': a.JSONify_dict(day_of_week_means),
#                'Hour_means': a.JSONify_dict(Hour_means)}
# with open('mean_dummys.json', 'w') as f:
#     json.dump(mean_dummys, f)

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


### Find the right features to use
#-- save reslts in a list in the form (r_squared, [feature list], (intercept, params))
results = [('r_squared', ('feature','list'), ('intercept', 'params')),]

## Create numpy arrays
values_array = training_data['ENTRIESn_hourly'].values
test_values_array = test_data['ENTRIESn_hourly'].values

### Test every variable independently against the
#-- List of features to explore
features_to_explore = ['UNIT', 'day_of_week', 'Hour', 'meanpressurei', 'fog', 'rain', 'meanwindspdi', 'meantempi', 'precipi', 'maxpressurei', 'maxdewpti', 'mintempi',  'mindewpti', 'minpressurei', 'meandewpti', 'maxtempi', 'UNIT_means', 'day_of_week_means', 'Hour_means']
for feature in features_to_explore:
    #-- extract feature
    if feature in ['UNIT', 'day_of_week', 'Hour']:
        #-- Use pandas.get_dummies for UNIT, day_of_week, Hour
        feature_array = pd.get_dummies(training_data[feature], prefix=feature)
        test_feature_array = pd.get_dummies(test_data[feature], prefix=feature)
        dot = np.dot
    else:
        feature_array = training_data[feature].values
        test_feature_array = test_data[feature].values
        def dot(a,b):return a*b

    #-- generate predictions
    intercept, params = a.OLS_linear_regression(feature_array, values_array)
    predictions = dot(test_feature_array, params) + intercept
    #-- calculate r** using backup data
    r_squared = a.compute_r_squared(test_values_array, predictions)
    #-- append results to list
    results.append((r_squared, ([feature],), (intercept, tuple(params.tolist()))))

# print ([[x[0], x[1][0][0]] for x in results])
    #>[['r_squared', 'f'],
    #  [0.41004488886519175, 'UNIT'],
    #  [0.013480119273690194, 'day_of_week'],
    #  [0.11918213646536135, 'Hour'],
    #  [4.6893252414248465e-05, 'meanpressurei'],
    #  [0.00010419321888721633, 'fog'],
    #  [-0.00016555909505555633, 'rain'],
    #  [9.5051826772496462e-05, 'meanwindspdi'],
    #  [0.0002848165517614909, 'meantempi'],
    #  [-0.00018163984433350322, 'precipi'],
    #  [9.0488266448751631e-06, 'maxpressurei'],
    #  [-0.0001768707542453285, 'maxdewpti'],
    #  [0.00058739536864838016, 'mintempi'],
    #  [-0.00018067147690326024, 'mindewpti'],
    #  [0.00028448751528775684, 'minpressurei'],
    #  [-0.00019278784489529244, 'meandewpti'],
    #  [-6.3230968838645651e-07, 'maxtempi'],
    #  [0.41547200266629658, 'UNIT_means'],
    #  [0.013506567080348253, 'day_of_week_means'],
    #  [0.11935876457150507, 'Hour_means']]

### Using dummy variables & OLS still does not result in good predictions
### Implementing SGD...

# for feature in features_to_explore:
#     #-- extract feature
#     if feature in ['UNIT', 'day_of_week', 'Hour']:
#         #-- Use pandas.get_dummies for UNIT, day_of_week, Hour
#         feature_array = pd.get_dummies(training_data[feature], prefix=feature)
#         test_feature_array = pd.get_dummies(test_data[feature], prefix=feature)
#         dot = np.dot
#     else:
#         feature_array = training_data[feature].values
#         test_feature_array = test_data[feature].values
#         def dot(a,b):return a*b
#
#     #-- generate predictions
#     intercept, params = a.SGD_regression(feature_array, values_array)
#     predictions = dot(test_feature_array, params) + intercept
#     #-- calculate r** using backup data
#     r_squared = a.compute_r_squared(test_values_array, predictions)
#     #-- append results to list
#     results.append((r_squared, ([feature],), (intercept, tuple(params.tolist()))))

# print ([[x[0], x[1][0][0]] for x in results])
    #>[['r_squared', 'f'],
    #  [0.35693821347008892, 'UNIT'],
    #  [-0.13336887490640703, 'day_of_week'],
    #  [0.054474268865662423, 'Hour']]

### Test the best variable with one other variable at a time
## Eliminate 'UNIT', 'day_of_week', 'Hour' from the possible features because the mean dummy units predict better.
## Remove UNIT_means and add it to all feature sets
features_to_explore = ['meanpressurei', 'fog', 'rain', 'meanwindspdi', 'meantempi', 'precipi', 'maxpressurei', 'maxdewpti', 'mintempi',  'mindewpti', 'minpressurei', 'meandewpti', 'maxtempi', 'day_of_week_means', 'Hour_means']

for feature in features_to_explore:
    features = [feature,'UNIT_means']
    #-- extract feature
    feature_array = training_data[features].values
    test_feature_array = test_data[features].values

    #-- generate predictions
    intercept, params = a.OLS_linear_regression(feature_array, values_array)
    predictions = np.dot(test_feature_array, params) + intercept
    #-- calculate r** using backup data
    r_squared = a.compute_r_squared(test_values_array, predictions)
    #-- append results to list
    results.append((r_squared, (features,), (intercept, tuple(params.tolist()))))

# print ([[x[0], x[1][0]] for x in results])
    #>[['r_squared', 'feature'],
    #  [0.41559594882953754, ['meanpressurei', 'UNIT_means']],
    #  [0.41558729341793865, ['fog', 'UNIT_means']],
    #  [0.4154506737493936, ['rain', 'UNIT_means']],
    #  [0.41601434244915769, ['meanwindspdi', 'UNIT_means']],
    #  [0.41603923560882017, ['meantempi', 'UNIT_means']],
    #  [0.41529264612541006, ['precipi', 'UNIT_means']],
    #  [0.41554291112944919, ['maxpressurei', 'UNIT_means']],
    #  [0.41563415696138684, ['maxdewpti', 'UNIT_means']],
    #  [0.41638196279951056, ['mintempi', 'UNIT_means']],
    #  [0.41587675542449065, ['mindewpti', 'UNIT_means']],
    #  [0.4158260985182054, ['minpressurei', 'UNIT_means']],
    #  [0.41576041540946318, ['meandewpti', 'UNIT_means']],
    #  [0.41571645197371865, ['maxtempi', 'UNIT_means']],
    #  [0.42824082804989616, ['day_of_week_means', 'UNIT_means']],
    #  [0.47968730654524339, ['Hour_means', 'UNIT_means']]]

### Test Hour_means & UNIT_means with one other variable at a time
## Remove UNIT_means & Hour_means and add them to all feature sets
features_to_explore = ['meanpressurei', 'fog', 'rain', 'meanwindspdi', 'meantempi', 'precipi', 'maxpressurei', 'maxdewpti', 'mintempi',  'mindewpti', 'minpressurei', 'meandewpti', 'maxtempi', 'day_of_week_means']

for feature in features_to_explore:
    features = [feature, 'UNIT_means', 'Hour_means']
    #-- extract feature
    feature_array = training_data[features].values
    test_feature_array = test_data[features].values

    #-- generate predictions
    intercept, params = a.OLS_linear_regression(feature_array, values_array)
    predictions = np.dot(test_feature_array, params) + intercept
    #-- calculate r** using backup data
    r_squared = a.compute_r_squared(test_values_array, predictions)
    #-- append results to list
    results.append((r_squared, (features,), (intercept, tuple(params.tolist()))))

# print ([[x[0], x[1][0]] for x in results])
    #>[['r_squared', 'feature'],
    #  [0.47973690168712868, ['meanpressurei', 'UNIT_means', 'Hour_means']],
    #  [0.47979460100406623, ['fog', 'UNIT_means', 'Hour_means']],
    #  [0.47967663503601432, ['rain', 'UNIT_means', 'Hour_means']],
    #  [0.48036013058982785, ['meanwindspdi', 'UNIT_means', 'Hour_means']],
    #  [0.4800944047778759, ['meantempi', 'UNIT_means', 'Hour_means']],
    #  [0.47955870644631726, ['precipi', 'UNIT_means', 'Hour_means']],
    #  [0.47967275545351939, ['maxpressurei', 'UNIT_means', 'Hour_means']],
    #  [0.47977675413800425, ['maxdewpti', 'UNIT_means', 'Hour_means']],
    #  [0.48037894348391585, ['mintempi', 'UNIT_means', 'Hour_means']],
    #  [0.47996890142867377, ['mindewpti', 'UNIT_means', 'Hour_means']],
    #  [0.47996153050185486, ['minpressurei', 'UNIT_means', 'Hour_means']],
    #  [0.47986210732408541, ['meandewpti', 'UNIT_means', 'Hour_means']],
    #  [0.47984846041148965, ['maxtempi', 'UNIT_means', 'Hour_means']],
    #  [0.49370326835188771, ['day_of_week_means', 'UNIT_means', 'Hour_means']]]

### Test UNIT_means, Hour_means & day_of_week_means with one other variable at a time
## Remove UNIT_means, Hour_means & day_of_week_means and add them to all feature sets
features_to_explore = ['meanpressurei', 'fog', 'rain', 'meanwindspdi', 'meantempi', 'precipi', 'maxpressurei', 'maxdewpti', 'mintempi',  'mindewpti', 'minpressurei', 'meandewpti', 'maxtempi']

for feature in features_to_explore:
    features = [feature, 'UNIT_means', 'Hour_means', 'day_of_week_means']
    #-- extract feature
    feature_array = training_data[features].values
    test_feature_array = test_data[features].values

    #-- generate predictions
    intercept, params = a.OLS_linear_regression(feature_array, values_array)
    predictions = np.dot(test_feature_array, params) + intercept
    #-- calculate r** using backup data
    r_squared = a.compute_r_squared(test_values_array, predictions)
    #-- append results to list
    results.append((r_squared, (features,), (intercept, tuple(params.tolist()))))

# print ([[x[0], x[1][0]] for x in results])
    #>[['r_squared', 'feature'],
    #  [0.49370744074028183, ['meanpressurei', 'UNIT_means', 'Hour_means', 'day_of_week_means']],
    #  [0.49372753865892172, ['fog', 'UNIT_means', 'Hour_means', 'day_of_week_means']],
    #  [0.49378270281459169, ['rain', 'UNIT_means', 'Hour_means', 'day_of_week_means']],
    #  [0.49372031991677812, ['meanwindspdi', 'UNIT_means', 'Hour_means', 'day_of_week_means']],
    #  [0.49394608711807597, ['meantempi', 'UNIT_means', 'Hour_means', 'day_of_week_means']],
    #  [0.4938495272449126,  ['precipi', 'UNIT_means', 'Hour_means', 'day_of_week_means']],
    #  [0.49367905881773144, ['maxpressurei', 'UNIT_means', 'Hour_means', 'day_of_week_means']],
    #  [0.49376195072109108, ['maxdewpti', 'UNIT_means', 'Hour_means', 'day_of_week_means']],
    #  [0.49389470802005497, ['mintempi', 'UNIT_means', 'Hour_means', 'day_of_week_means']],
    #  [0.49373900639359747, ['mindewpti', 'UNIT_means', 'Hour_means', 'day_of_week_means']],
    #  [0.49378156919660487, ['minpressurei', 'UNIT_means', 'Hour_means', 'day_of_week_means']],
    #  [0.49373445063798294, ['meandewpti', 'UNIT_means', 'Hour_means', 'day_of_week_means']],
    #  [0.49397803935218865, ['maxtempi', 'UNIT_means', 'Hour_means', 'day_of_week_means']]]

## Test adding maxtempi
features_to_explore = ['meanpressurei', 'fog', 'rain', 'meanwindspdi', 'meantempi', 'precipi', 'maxpressurei', 'maxdewpti', 'mintempi',  'mindewpti', 'minpressurei', 'meandewpti']

for feature in features_to_explore:
    features = [feature, 'UNIT_means', 'Hour_means', 'day_of_week_means', 'maxtempi']
    #-- extract feature
    feature_array = training_data[features].values
    test_feature_array = test_data[features].values

    #-- generate predictions
    intercept, params = a.OLS_linear_regression(feature_array, values_array)
    predictions = np.dot(test_feature_array, params) + intercept
    #-- calculate r** using backup data
    r_squared = a.compute_r_squared(test_values_array, predictions)
    #-- append results to list
    results.append((r_squared, (features,), (intercept, tuple(params.tolist()))))

# print ([[x[0], x[1][0]] for x in results])
    #>[['r_squared', 'feature'],
    #  [0.49397619378619528, ['meanpressurei', 'UNIT_means', 'Hour_means', 'day_of_week_means', 'maxtempi']],
    #  [0.49399801652926134, ['fog', 'UNIT_means', 'Hour_means', 'day_of_week_means', 'maxtempi']],
    #  [0.49415940953240234, ['rain', 'UNIT_means', 'Hour_means', 'day_of_week_means', 'maxtempi']],
    #  [0.49397862264605163, ['meanwindspdi', 'UNIT_means', 'Hour_means', 'day_of_week_means', 'maxtempi']],
    #  [0.49395867822271988, ['meantempi', 'UNIT_means', 'Hour_means', 'day_of_week_means', 'maxtempi']],
    #  [0.49430849100089047, ['precipi', 'UNIT_means', 'Hour_means', 'day_of_week_means', 'maxtempi']],
    #  [0.49394595436532107, ['maxpressurei', 'UNIT_means', 'Hour_means', 'day_of_week_means', 'maxtempi']],
    #  [0.49397478619625901, ['maxdewpti', 'UNIT_means', 'Hour_means', 'day_of_week_means', 'maxtempi']],
    #  [0.49397076793747408, ['mintempi', 'UNIT_means', 'Hour_means', 'day_of_week_means', 'maxtempi']],
    #  [0.49394704683251178, ['mindewpti', 'UNIT_means', 'Hour_means', 'day_of_week_means', 'maxtempi']],
    #  [0.49404619381526171, ['minpressurei', 'UNIT_means', 'Hour_means', 'day_of_week_means', 'maxtempi']],
    #  [0.49395595938817671, ['meandewpti', 'UNIT_means', 'Hour_means', 'day_of_week_means', 'maxtempi']]]

# Test adding precipi
features_to_explore = ['meanpressurei', 'fog', 'rain', 'meanwindspdi', 'meantempi', 'maxpressurei', 'maxdewpti', 'mintempi',  'mindewpti', 'minpressurei', 'meandewpti']

for feature in features_to_explore:
    features = [feature, 'UNIT_means', 'Hour_means', 'day_of_week_means', 'maxtempi', 'precipi']
    #-- extract feature
    feature_array = training_data[features].values
    test_feature_array = test_data[features].values

    #-- generate predictions
    intercept, params = a.OLS_linear_regression(feature_array, values_array)
    predictions = np.dot(test_feature_array, params) + intercept
    #-- calculate r** using backup data
    r_squared = a.compute_r_squared(test_values_array, predictions)
    #-- append results to list
    results.append((r_squared, (features,), (intercept, tuple(params.tolist()))))

# print ([[x[0], x[1][0]] for x in results])
    #>[['r_squared', 'feature'],
    #  [0.49435730653077181, ['meanpressurei', 'UNIT_means', 'Hour_means', 'day_of_week_means', 'maxtempi', 'precipi']],
    #  [0.49460171140718767, ['fog', 'UNIT_means', 'Hour_means', 'day_of_week_means', 'maxtempi', 'precipi']],
    #  [0.4942978801376432, ['rain', 'UNIT_means', 'Hour_means', 'day_of_week_means', 'maxtempi', 'precipi']],
    #  [0.49431046392161515, ['meanwindspdi', 'UNIT_means', 'Hour_means', 'day_of_week_means', 'maxtempi', 'precipi']],
    #  [0.49427944872247853, ['meantempi', 'UNIT_means', 'Hour_means', 'day_of_week_means', 'maxtempi', 'precipi']],
    #  [0.4942937903172262, ['maxpressurei', 'UNIT_means', 'Hour_means', 'day_of_week_means', 'maxtempi', 'precipi']],
    #  [0.49433418773417048, ['maxdewpti', 'UNIT_means', 'Hour_means', 'day_of_week_means', 'maxtempi', 'precipi']],
    #  [0.49427944658657152, ['mintempi', 'UNIT_means', 'Hour_means', 'day_of_week_means', 'maxtempi', 'precipi']],
    #  [0.49424410869396584, ['mindewpti', 'UNIT_means', 'Hour_means', 'day_of_week_means', 'maxtempi', 'precipi']],
    #  [0.49442291041055964, ['minpressurei', 'UNIT_means', 'Hour_means', 'day_of_week_means', 'maxtempi', 'precipi']],
    #  [0.49427915776113918, ['meandewpti', 'UNIT_means', 'Hour_means', 'day_of_week_means', 'maxtempi', 'precipi']]]

# Test adding fog compare with 0.49460171140718767
features_to_explore = ['meanpressurei', 'rain', 'meanwindspdi', 'meantempi', 'maxpressurei', 'maxdewpti', 'mintempi',  'mindewpti', 'minpressurei', 'meandewpti']

for feature in features_to_explore:
    features = [feature, 'UNIT_means', 'Hour_means', 'day_of_week_means', 'maxtempi', 'precipi', 'fog']
    #-- extract feature
    feature_array = training_data[features].values
    test_feature_array = test_data[features].values

    #-- generate predictions
    intercept, params = a.OLS_linear_regression(feature_array, values_array)
    predictions = np.dot(test_feature_array, params) + intercept
    #-- calculate r** using backup data
    r_squared = a.compute_r_squared(test_values_array, predictions)
    #-- append results to list
    results.append((r_squared, (features,), (intercept, tuple(params.tolist()))))

# print ([[x[0], x[1][0]] for x in results])
    #>[['r_squared', 'feature'],
    #  [0.49459748045790541, ['meanpressurei', 'UNIT_means', 'Hour_means', 'day_of_week_means', 'maxtempi', 'precipi', 'fog']],
    #  [0.49465295145307331, ['rain', 'UNIT_means', 'Hour_means', 'day_of_week_means', 'maxtempi', 'precipi', 'fog']],
    #  [0.49462377994181617, ['meanwindspdi', 'UNIT_means', 'Hour_means', 'day_of_week_means', 'maxtempi', 'precipi', 'fog']],
    #  [0.49456993308680874, ['meantempi', 'UNIT_means', 'Hour_means', 'day_of_week_means', 'maxtempi', 'precipi', 'fog']],
    #  [0.49457760432961051, ['maxpressurei', 'UNIT_means', 'Hour_means', 'day_of_week_means', 'maxtempi', 'precipi', 'fog']],
    #  [0.49458594651000443, ['maxdewpti', 'UNIT_means', 'Hour_means', 'day_of_week_means', 'maxtempi', 'precipi', 'fog']],
    #  [0.49458519727104444, ['mintempi', 'UNIT_means', 'Hour_means', 'day_of_week_means', 'maxtempi', 'precipi', 'fog']],
    #  [0.494524766286221, ['mindewpti', 'UNIT_means', 'Hour_means', 'day_of_week_means', 'maxtempi', 'precipi', 'fog']],
    #  [0.49461007182343686, ['minpressurei', 'UNIT_means', 'Hour_means', 'day_of_week_means', 'maxtempi', 'precipi', 'fog']],
    #  [0.49454481157040175, ['meandewpti', 'UNIT_means', 'Hour_means', 'day_of_week_means', 'maxtempi', 'precipi', 'fog']]]

# Test adding rain compare with 0.49465295145307331
features_to_explore = ['meanpressurei', 'meanwindspdi', 'meantempi', 'maxpressurei', 'maxdewpti', 'mintempi',  'mindewpti', 'minpressurei', 'meandewpti']

for feature in features_to_explore:
    features = [feature, 'UNIT_means', 'Hour_means', 'day_of_week_means', 'maxtempi', 'precipi', 'fog', 'rain']
    #-- extract feature
    feature_array = training_data[features].values
    test_feature_array = test_data[features].values

    #-- generate predictions
    intercept, params = a.OLS_linear_regression(feature_array, values_array)
    predictions = np.dot(test_feature_array, params) + intercept
    #-- calculate r** using backup data
    r_squared = a.compute_r_squared(test_values_array, predictions)
    #-- append results to list
    results.append((r_squared, (features,), (intercept, tuple(params.tolist()))))

# print ([[x[0], x[1][0]] for x in results])
    #>[['r_squared', 'feature'],
    #  [0.4946525609002429, ['meanpressurei', 'UNIT_means', 'Hour_means', 'day_of_week_means', 'maxtempi', 'precipi', 'fog', 'rain']],
    #  [0.49466806849051093, ['meanwindspdi', 'UNIT_means', 'Hour_means', 'day_of_week_means', 'maxtempi', 'precipi', 'fog', 'rain']],
    #  [0.49462569628221476, ['meantempi', 'UNIT_means', 'Hour_means', 'day_of_week_means', 'maxtempi', 'precipi', 'fog', 'rain']],
    #  [0.49462586265978714, ['maxpressurei', 'UNIT_means', 'Hour_means', 'day_of_week_means', 'maxtempi', 'precipi', 'fog', 'rain']],
    #  [0.49469115125765351, ['maxdewpti', 'UNIT_means', 'Hour_means', 'day_of_week_means', 'maxtempi', 'precipi', 'fog', 'rain']],
    #  [0.4946351051575969, ['mintempi', 'UNIT_means', 'Hour_means', 'day_of_week_means', 'maxtempi', 'precipi', 'fog', 'rain']],
    #  [0.49461647391320085, ['mindewpti', 'UNIT_means', 'Hour_means', 'day_of_week_means', 'maxtempi', 'precipi', 'fog', 'rain']],
    #  [0.49466634958996603, ['minpressurei', 'UNIT_means', 'Hour_means', 'day_of_week_means', 'maxtempi', 'precipi', 'fog', 'rain']],
    #  [0.49464460779303077, ['meandewpti', 'UNIT_means', 'Hour_means', 'day_of_week_means', 'maxtempi', 'precipi', 'fog', 'rain']]]

### Try it with all features to see if it makes a difference.
features = ['UNIT_means', 'Hour_means', 'day_of_week_means', 'meanpressurei', 'fog', 'rain', 'meanwindspdi', 'meantempi', 'precipi', 'maxpressurei', 'maxdewpti', 'mintempi',  'mindewpti', 'minpressurei', 'meandewpti', 'maxtempi']

#-- extract feature
feature_array = training_data[features].values
test_feature_array = test_data[features].values

#-- generate predictions
intercept, params = a.OLS_linear_regression(feature_array, values_array)
predictions = np.dot(test_feature_array, params) + intercept
#-- calculate r** using backup data
r_squared = a.compute_r_squared(test_values_array, predictions)
#-- append results to list
results.append((r_squared, (features,), (intercept, tuple(params.tolist()))))

# print ([[x[0], x[1][0]] for x in results])
    #>[['r_squared', 'feature'],
    #  [0.4953608732252669, ['UNIT_means', 'Hour_means', 'day_of_week_means', 'meanpressurei', 'fog', 'rain', 'meanwindspdi', 'meantempi', 'precipi', 'maxpressurei', 'maxdewpti', 'mintempi', 'mindewpti', 'minpressurei', 'meandewpti', 'maxtempi']]]

# Save results to json
with open('results.json', 'w') as f:
    json.dump(results, f)
