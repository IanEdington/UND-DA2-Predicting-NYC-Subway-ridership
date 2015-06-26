'''
    1: Just try the plot equation
    2: how to filter by rain ref: problem set 2: 7 filter
    3: how to plot data
    4: making the data look better:
        http://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes.hist
'''

import numpy as np
import pandas
import matplotlib.pyplot as plt

def entries_histogram(turnstile_weather):
    '''
    Before we perform any analysis, it might be useful to take a
    look at the data we're hoping to analyze. More specifically, let's
    examine the hourly entries in our NYC subway data and determine what
    distribution the data follows. This data is stored in a dataframe
    called turnstile_weather under the ['ENTRIESn_hourly'] column.

    Let's plot two histograms on the same axes to show hourly
    entries when raining vs. when not raining. Here's an example on how
    to plot histograms with pandas and matplotlib:
    turnstile_weather['column_to_graph'].hist()

    Your histograph may look similar to bar graph in the instructor notes below.

    You can read a bit about using matplotlib and pandas to plot histograms here:
    http://pandas.pydata.org/pandas-docs/stable/visualization.html#histograms

    You can see the information contained within the turnstile weather data here:
    https://www.dropbox.com/s/meyki2wl9xfa7yk/turnstile_data_master_with_weather.csv
    '''

    plt.figure()

    turnstile_weather[turnstile_weather.rain==0]['ENTRIESn_hourly'].hist(range=(0,6000), stacked=True, label='No Rain') # your code here to plot a historgram for hourly entries when it is not raining
    turnstile_weather[turnstile_weather.rain==1]['ENTRIESn_hourly'].hist(range=(0,6000), stacked=True, label='Rain') # your code here to plot a historgram for hourly entries when it is raining

    return plt


'''
    1: start with means of each - worked correctly
    2: now the hard part ;P Mann-Whitney
    Everything worked correctly
'''

import numpy as np
import scipy
import scipy.stats
import pandas

def mann_whitney_plus_means(turnstile_weather):
    '''
    This function will consume the turnstile_weather dataframe containing
    our final turnstile weather data.

    You will want to take the means and run the Mann Whitney U-test on the
    ENTRIESn_hourly column in the turnstile_weather dataframe.

    This function should return:
        1) the mean of entries with rain
        2) the mean of entries without rain
        3) the Mann-Whitney U-statistic and p-value comparing the number of entries
           with rain and the number of entries without rain

    You should feel free to use scipy's Mann-Whitney implementation, and you
    might also find it useful to use numpy's mean function.

    Here are the functions' documentation:
    http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html
    http://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html

    You can look at the final turnstile weather data at the link below:
    https://www.dropbox.com/s/meyki2wl9xfa7yk/turnstile_data_master_with_weather.csv
    '''

    '''
    This function should return:
        1) the mean of entries with rain
        2) the mean of entries without rain
        3) the Mann-Whitney U-statistic and p-value comparing the number of entries
           with rain and the number of entries without rain
    '''

    with_rain = turnstile_weather[turnstile_weather.rain == 1]['ENTRIESn_hourly']
    without_rain = turnstile_weather[turnstile_weather.rain == 0]['ENTRIESn_hourly']

    with_rain_mean = with_rain.mean()
    without_rain_mean = without_rain.mean()

    U,p =scipy.stats.mannwhitneyu(with_rain, without_rain)

    return with_rain_mean, without_rain_mean, U, p # leave this line for the grader


'''
from linear_regression exercise:
    iter1: what is given?
    iter2: copy tutorial to see if I can reproduce results
        http://statsmodels.sourceforge.net/0.5.0/generated/statsmodels.regression.linear_model.OLS.html
    iter3: is working now I just have to figure out how to make it give the right answer
    iter4: changed values and features (x,y)
    iter5: how do you get the intercepts?
        http://statsmodels.sourceforge.net/0.5.0/generated/statsmodels.regression.linear_model.OLS.fit.html
        http://statsmodels.sourceforge.net/0.5.0/generated/statsmodels.regression.linear_model.RegressionResults.html
        found intercept by reading instructions
    iter6: formating results for answer

problem set:
    iter1: test linear_regression
'''

import numpy as np
import pandas
import statsmodels.api as sm

"""
In this question, you need to:
1) implement the linear_regression() procedure
2) Select features (in the predictions procedure) and make predictions.

"""

def linear_regression(features, values):
    """
    Perform linear regression given a data set with an arbitrary number of features.

    This can be the same code as in the lesson #3 exercise.
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
    dummy_units = pandas.get_dummies(dataframe['UNIT'], prefix='unit')
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

'''
    1: use ENTRIESn_hourly for the comparison against the prediction
        because that's what you are trying to predict.
        -> delta seems normally distributed
    2: try different bin sizes
        -> bin = 20
    3: limit range
        -> range = (-10000, 10000)
'''
import numpy as np
import scipy
import matplotlib.pyplot as plt

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

import numpy as np
import scipy
import matplotlib.pyplot as plt
import sys

def compute_r_squared(data, predictions):
    '''
    In exercise 5, we calculated the R^2 value for you. But why don't you try and
    and calculate the R^2 value yourself.

    Given a list of original data points, and also a list of predicted data points,
    write a function that will compute and return the coefficient of determination (R^2)
    for this data.  numpy.mean() and numpy.sum() might both be useful here, but
    not necessary.

    Documentation about numpy.mean() and numpy.sum() below:
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

from pandas import *
from ggplot import *

def plot_weather_data(turnstile_weather):
    '''
    You are passed in a dataframe called turnstile_weather.
    Use turnstile_weather along with ggplot to make a data visualization
    focused on the MTA and weather data we used in assignment #3.
    You should feel free to implement something that we discussed in class
    (e.g., scatterplots, line plots, or histograms) or attempt to implement
    something more advanced if you'd like.

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

    plot = # your code here
    return plot

from pandas import *
from ggplot import *

def plot_weather_data(turnstile_weather):
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

    plot = # your code here
    return plot
