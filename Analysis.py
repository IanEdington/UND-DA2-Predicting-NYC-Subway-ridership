# Reference:
#   http://pandas.pydata.org/pandas-docs/stable/visualization.html#histograms
#   http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html
#   http://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html
        http://statsmodels.sourceforge.net/0.5.0/generated/statsmodels.regression.linear_model.OLS.html
        http://statsmodels.sourceforge.net/0.5.0/generated/statsmodels.regression.linear_model.OLS.fit.html
        http://statsmodels.sourceforge.net/0.5.0/generated/statsmodels.regression.linear_model.RegressionResults.html
    http://www.itl.nist.gov/div898/handbook/pri/section2/pri24.htm
        http://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html
        http://docs.scipy.org/doc/numpy/reference/generated/numpy.sum.html
    https://pypi.python.org/pypi/ggplot/



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
#import scipy.stats
import statsmodels.api as sm
import sys
#from ggplot import *

def r_nor_entries_histogram(turnstile_weather):
    '''
    consume: the turnstile_weather dataframe

    return: Two histograms on the same axes to show hourly entries when raining vs. when not raining.

    Reference:
        http://pandas.pydata.org/pandas-docs/stable/visualization.html#histograms
    '''

    plt.figure()

    turnstile_weather[turnstile_weather.rain==0]['ENTRIESn_hourly'].hist(range=(0,6000), stacked=True, label='No Rain')
    turnstile_weather[turnstile_weather.rain==1]['ENTRIESn_hourly'].hist(range=(0,6000), stacked=True, label='Rain')

    return plt

def mann_whitney_plus_means(turnstile_weather):
    '''
    consume: the turnstile_weather dataframe

    return: 1, 2, 3, 4
        1) the mean of entries with rain
        2) the mean of entries without rain
        3) the Mann-Whitney U-statistic and
        4) p-value comparing the number of entries with rain and the number of entries without rain

    Reference:
        http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html
        http://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html
    '''

    with_rain = turnstile_weather[turnstile_weather.rain == 1]['ENTRIESn_hourly']
    without_rain = turnstile_weather[turnstile_weather.rain == 0]['ENTRIESn_hourly']

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

def predictions(dataframe):
    '''
    The NYC turnstile data is stored in a pandas dataframe called weather_turnstile.
    Using the information stored in the dataframe, let's predict the ridership of
    the NYC subway using linear regression with gradient descent.

    You can download the complete turnstile weather dataframe here:
    https://www.dropbox.com/s/meyki2wl9xfa7yk/turnstile_data_master_with_weather.csv

    Your prediction should have a R^2 value of 0.40 or better.
    You need to experiment using various input features contained in the dataframe.
    We recommend that you don't use the EXITSn_hourly feature as an input to the
    linear model because we cannot use it as a predictor: we cannot use exits
    counts as a way to predict entry counts.

    Note: Due to the memory and CPU limitation of our Amazon EC2 instance, we will
    give you a random subet (~10%) of the data contained in
    turnstile_data_master_with_weather.csv. You are encouraged to experiment with
    this exercise on your own computer, locally. If you do, you may want to complete Exercise
    least squares can be very slow for a large number of features.

    If you receive a "server has encountered an error" message, that means you are
    hitting the 30-second limit that's placed on running your program. Try using a
    smaller number of features.
    '''
    #* 0.478188671128 = ['Hour']
    #* 0.443493884388 = ['mintempi']
    #* 0.443213813855 = ['meantempi']
    #* 0.443165780482 = ['meanwindspdi']
    #* 0.442914509098 = ['mindewpti']
    #* 0.442836865996 = ['minpressurei']
    #  0.442799206472 = ['maxtempi']
    #  0.44279446139  = ['fog']
    #  0.442634404331 = ['meandewpti']
    #  0.442583734674 = ['meanpressurei']
    #  0.442434339041 = ['maxdewpti']
    #  0.442453874032 = ['maxpressurei']
    #  0.442423106498 = ['rain']
    #  0.442423593893 = ['precipi']
    #  error = ['thunder'] (all zero's)

    # 0.47924770782  = ['rain', 'precipi',  'meantempi']
    # 0.479279488045 = ['rain', 'precipi', 'Hour', 'meantempi', 'meandewpti' ]
    # 0.479687594832 = ['rain', 'precipi', 'Hour', 'meantempi', 'meanwindspdi' ]
    # 0.479573235615 = ['rain', 'precipi', 'Hour', 'meantempi', 'meanpressurei' ]
    # 0.444588458149 = ['mintempi', 'meanwindspdi', 'mindewpti', 'minpressurei' ]
    # 0.444596465729 = ['rain', 'mintempi', 'meanwindspdi', 'mindewpti', 'minpressurei' ]
    # 0.480650382098 = ['Hour', 'mintempi', 'meanwindspdi', 'mindewpti', 'minpressurei' ]
    # 0.480308674906 = ['Hour', 'meantempi', 'meanwindspdi', 'mindewpti', 'minpressurei' ]
    # 0.481031760379 = ['Hour', 'mintempi', 'meanwindspdi', 'mindewpti', 'minpressurei', 'fog']
    # 0.480657821706 = ['Hour', 'mintempi', 'meanwindspdi', 'mindewpti', 'minpressurei', 'rain' ]
    # 0.481046407683 = ['Hour', 'mintempi', 'meanwindspdi', 'mindewpti', 'minpressurei', 'fog', 'rain' ]
    # 0.481366117136 = ['Hour', 'mintempi', 'meanwindspdi', 'mindewpti', 'minpressurei', 'fog', 'meantempi' ]
    # 0.481579983337 = ['Hour', 'mintempi', 'meantempi', 'meanwindspdi', 'mindewpti', 'minpressurei', 'maxtempi', 'fog' ]


    feature_list = ['Hour', 'mintempi', 'meantempi', 'meanwindspdi', 'mindewpti', 'minpressurei', 'maxtempi', 'fog' ]


    # Select Features (try different features!)
    features = dataframe[feature_list]

    # Add UNIT to features using dummy variables
    dummy_units = pd.get_dummies(dataframe['UNIT'], prefix='unit')
    features = features.join(dummy_units)

    # Values
    values = dataframe['ENTRIESn_hourly']

    # Get the numpy arrays
    features_array = features.values
    values_array = values.values

    # Perform linear regression
    intercept, params = linear_regression(features_array, values_array)

    predictions = intercept + np.dot(features_array, params)
    return predictions

def plot_residuals(turnstile_weather, predictions):
    '''
    Using the same methods that we used to plot a histogram of entries
    per hour for our data, why don't you make a histogram of the residuals
    (that is, the difference between the original hourly entry data and the predicted values).
    Try different binwidths for your histogram.

    Based on this residual histogram, do you have any insight into how our model
    performed?  Reading a bit on this webpage might be useful:

    http://www.itl.nist.gov/div898/handbook/pri/section2/pri24.htm
    '''

    plt.figure()
    (turnstile_weather['ENTRIESn_hourly'] - predictions).hist(bins=20, range=(-10000,10000))
    return plt

'''
From Lesson 3: Calculating R^2
    1: how is the data formated?
    2: testing out how df's work with oporators
    3: implementing equation
Now:
    1: test if equation still works
    2: break it down explaining the steps
'''

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

#def plot_weather_data(turnstile_weather):
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

#def plot_weather_data(turnstile_weather):
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
