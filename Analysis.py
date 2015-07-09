# Reference:
#   http://pandas.pydata.org/pandas-docs/stable/visualization.html#histograms
#   http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html
#   http://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html
#    http://statsmodels.sourceforge.net/0.5.0/generated/statsmodels.regression.linear_model.OLS.html
#    http://statsmodels.sourceforge.net/0.5.0/generated/statsmodels.regression.linear_model.OLS.fit.html
#    http://statsmodels.sourceforge.net/0.5.0/generated/statsmodels.regression.linear_model.RegressionResults.html
#    http://www.itl.nist.gov/div898/handbook/pri/section2/pri24.htm
#    http://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html
#    http://docs.scipy.org/doc/numpy/reference/generated/numpy.sum.html
#    https://pypi.python.org/pypi/ggplot/



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
#import scipy.stats
import statsmodels.api as sm
import sys
#from ggplot import *

def read_csv():
    data = pd.read_csv(r'turnstile_data_master_with_weather.csv')
        # data = pd.read_csv(r'turnstile_weather_v2.csv')
    return data

def r_nor_entries_histogram(data):
    '''
    consume: the turnstile_weather dataframe

    return: Two histograms on the same axes to show hourly entries when raining vs. when not raining.

    Reference:
        http://pandas.pydata.org/pandas-docs/stable/visualization.html#histograms
    '''

    plt.figure()

    data[data.rain==0]['ENTRIESn_hourly'].hist(range=(0,6000), stacked=True, label='No Rain')
    data[data.rain==1]['ENTRIESn_hourly'].hist(range=(0,6000), stacked=True, label='Rain')

    return plt

def mann_whitney_plus_means(df1, df2):
    '''
    consume: two filtered turnstile_weather dataframes that you want to compare
        ie: data[data.rain == 1]
            data[data.rain == 0]

    return: 1, 2, 3, 4
        1) the mean of entries of df1
        2) the mean of entries of df2
        3) the Mann-Whitney U-statistic and
        4) p-value comparing entries from df1 and entries from df2

    Reference:
        http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html
        http://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html
    '''

    with_rain = df1['ENTRIESn_hourly']
    without_rain = df2['ENTRIESn_hourly']

    with_rain_mean = with_rain.mean()
    without_rain_mean = without_rain.mean()

    U,p =scipy.stats.mannwhitneyu(with_rain, without_rain)

    return with_rain_mean, without_rain_mean, U, p # leave this line for the grader

def linear_regression(features, values):
    """
    Perform linear regression given a data set with an arbitrary number of features.

    Reference:
        http://statsmodels.sourceforge.net/0.5.0/generated/statsmodels.regression.linear_model.OLS.html
        http://statsmodels.sourceforge.net/0.5.0/generated/statsmodels.regression.linear_model.OLS.fit.html
        http://statsmodels.sourceforge.net/0.5.0/generated/statsmodels.regression.linear_model.RegressionResults.html
    """

    features = sm.add_constant(features)
    model = sm.OLS(values, features)
    results = model.fit()
    params = results.params

    intercept = params[0]
    params = params[1:]

    return intercept, params

def predictions(data, feature_list = ['Hour', 'mintempi', 'meantempi', 'meanwindspdi', 'mindewpti', 'minpressurei', 'maxtempi', 'fog' ]):

    '''
    Using the information stored in the dataframe, let's predict the ridership of
    the NYC subway using linear regression with gradient descent.
    '''

    # Select Features
    features = data[feature_list]

    # Add UNIT to features using dummy variables
    dummy_units = pd.get_dummies(data['UNIT'], prefix='unit')
    features = features.join(dummy_units)

    # Values
    values = data['ENTRIESn_hourly']

    # Get the numpy arrays
    features_array = features.values
    values_array = values.values

    # Perform linear regression
    intercept, params = linear_regression(features_array, values_array)

    predictions = intercept + np.dot(features_array, params)
    return predictions

def plot_residuals(data, predictions):
    '''
    http://www.itl.nist.gov/div898/handbook/pri/section2/pri24.htm
    '''

    plt.figure()
    (data['ENTRIESn_hourly'] - predictions).hist(bins=20, range=(-10000,10000))
    return plt

def compute_r_squared(data, predictions):
    '''
    consume: numpy list of original data points, numpy list of predicted data points
    return: the coefficient of determination (R^2)

    Reference:
        http://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html
        http://docs.scipy.org/doc/numpy/reference/generated/numpy.sum.html
    '''

    ### broken down ###
    # numerator = ((data-predictions)**2).sum()
    # denominator = ((data-np.mean(data))**2).sum()
    # r_squared = 1 - numerator/denominator
    # return r_squared

    # one liner equation
    return 1 - (((data-predictions)**2).sum())/(((data-np.mean(data))**2).sum())

###visualization

#def plot_weather_data(data):
    '''
    scatterplots, line plots, or histograms

    Here are some suggestions for things to investigate and illustrate:
     * Ridership by time of day or day of week
     * How ridership varies based on Subway station (UNIT)
     * Which stations have more exits or entries at different times of day
       (You can use UNIT as a proxy for subway station.)

    If you'd like to learn more about ggplot and its capabilities, take
    a look at the documentation at:
    https://pypi.python.org/pypi/ggplot/

    You can check out:
    https://www.dropbox.com/s/meyki2wl9xfa7yk/turnstile_data_master_with_weather.csv

    To see all the columns and data points included in the turnstile_weather
    dataframe.

    However, due to the limitation of our Amazon EC2 server, we are giving you a random
    subset, about 1/3 of the actual data in the turnstile_weather dataframe.
    '''

#    plot = # your code here
#    return plot

#def plot_weather_data(data):
    '''
    plot_weather_data is passed a dataframe called turnstile_weather.
    Use turnstile_weather along with ggplot to make another data visualization
    focused on the MTA and weather data we used in Project 3.

    Make a type of visualization different than what you did in the previous exercise.
    Try to use the data in a different way (e.g., if you made a lineplot concerning
    ridership and time of day in exercise #1, maybe look at weather and try to make a
    histogram in this exercise). Or try to use multiple encodings in your graph if
    you didn't in the previous exercise.

    You should feel free to implement something that we discussed in class
    (e.g., scatterplots, line plots, or histograms) or attempt to implement
    something more advanced if you'd like.

    Here are some suggestions for things to investigate and illustrate:
     * Ridership by time-of-day or day-of-week
     * How ridership varies by subway station (UNIT)
     * Which stations have more exits or entries at different times of day
       (You can use UNIT as a proxy for subway station.)

    If you'd like to learn more about ggplot and its capabilities, take
    a look at the documentation at:
    https://pypi.python.org/pypi/ggplot/

    You can check out the link
    https://www.dropbox.com/s/meyki2wl9xfa7yk/turnstile_data_master_with_weather.csv
    to see all the columns and data points included in the turnstile_weather
    dataframe.

   However, due to the limitation of our Amazon EC2 server, we are giving you a random
    subset, about 1/3 of the actual data in the turnstile_weather dataframe.
    '''

    #plot = # your code here
    #return plot
