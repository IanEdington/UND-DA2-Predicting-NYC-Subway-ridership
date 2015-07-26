import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import scipy.stats
# import statsmodels.api as sm
import sys
from sklearn.linear_model import SGDRegressor
from itertools import combinations


def bar_plot_mean_Entries(data, feature, variable = 'ENTRIESn_hourly'):
    '''
    bar chart of ENTRIESn_hourly by UNIT
    http://pandas.pydata.org/pandas-docs/stable/groupby.html
    http://wiki.scipy.org/Cookbook/Matplotlib/BarCharts
    '''
    gby_UNIT_mean = data[[feature, 'ENTRIESn_hourly']].groupby(feature, as_index=False).mean()
    x_axis = gby_UNIT_mean[feature]
    y_axis = gby_UNIT_mean['ENTRIESn_hourly']

    # Ploting and saving figure
    plt.figure()

    if feature in ['UNIT', 'DATEn']:
        xlocations = np.array(range(len(y_axis))) + 0.5
        width = 0.5
        plt.bar(xlocations, y_axis, width=width)
        title = 'BAR plot of '+feature+' vs mean('+variable+')'
    else:
        plt.scatter(x_axis, y_axis)
        title = 'Scatter plot of '+feature+' vs mean('+variable+')'

    plt.title(title)
    plt.xlabel(feature)
    plt.ylabel(variable)
    plt.axis('tight')
    plt.savefig(title+'.png', bbox_inches='tight')
    plt.close('all')

def hist_MWW_suitability(series1, series2, rORf='fog'):
    '''
    Plots two histograms to determin if the Mann-Whitney U-test (MWW) is an appropiate statistical method. If the two overlapping Histograms are of similar distribution one of MWW's assumptions is met.

    consume: the turnstile_weather dataframe

    return: Two histograms on the same axes to show hourly entries when raining vs. when not raining.

    Reference:
        http://pandas.pydata.org/pandas-docs/stable/visualization.html#histograms
    '''

    plt.figure()

    if rORf=='rain':
        series1.hist(bins=12, range=(0,6000), color='Blue', label='No Rain', stacked=True)
        series2.hist(bins=12, range=(0,6000), color='Green', label='Rain', stacked=True)
        title = 'Volume of ridership by whether or not it is raining'
    else:
        series1.hist(bins=12, range=(0,6000), color='Blue', label='No Fog', stacked=True)
        series2.hist(bins=12, range=(0,6000), color='Green', label='Fog', stacked=True)
        title = 'Volume of ridership by whether or not there is fog'

    plt.legend(prop={'size': 14})
    plt.xlabel('volume of ridership')
    plt.ylabel('frequency of occurence')
    plt.ylim((0,16000))
    plt.title(title)
    return plt

def mann_whitney_plus_means(series1, series2):
    '''
    consume: two series that you want to compare
        ie: data[data.rain == 1]['ENTRIESn_hourly']
            data[data.rain == 0]['ENTRIESn_hourly']

    return: 1, 2, 3, 4
        1) the mean of entries of df1
        2) the mean of entries of df2
        3) the Mann-Whitney U-statistic and
        4) p-value comparing entries from df1 and entries from df2

    Reference:
        http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html
        http://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html
    '''

    U,p =scipy.stats.mannwhitneyu(series1, series2)

    return series1.mean(), series2.mean(), U, p*2

def are_features_perfectly_colinearity(data, features):
    stations = {}
    for index, row in data.iterrows():
        if row[features[0]] in stations:
            if stations[row[features[0]]] != row[features].tolist():
                return index, row, stations
        else:
            stations[row[features[0]]] = row[features].tolist()
    return 'perfectly colinear'

def OLS_linear_regression(features, values):
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

def compute_r_squared(values, predictions):
    '''
    consume: numpy list of original data points, numpy list of predicted data points
    return: the coefficient of determination (R^2)

    Reference:
        http://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html
        http://docs.scipy.org/doc/numpy/reference/generated/numpy.sum.html
    '''

    return 1 - (((values-predictions)**2).sum())/(((values-np.mean(values))**2).sum())

def feature_testing(data, all_features):
    results = {}
    # http://stackoverflow.com/questions/464864/python-code-to-pick-out-all-possible-combinations-from-a-list
    for L in range(1, len(stuff)+1):
        for features in combinations(all_features, L):

            #-- generate predictions
            feature_array = data[feature].values
            intercept, params = OLS_linear_regression(feature_array, values_array)

            #-- calculate r** using backup data
            test_feature_array = test_data[feature].values
            predictions = test_feature_array * params + intercept
            r_squared = compute_r_squared(test_values_array, predictions)
            #-- append results to list
            results.append((r_squared, ([feature],), (intercept, tuple(params.tolist()))))
    return results

#####################
### Visualization ###
#####################

def plot_residuals(data, predictions):
    '''
    http://www.itl.nist.gov/div898/handbook/pri/section2/pri24.htm
    '''

    plt.figure()
    (data['ENTRIESn_hourly'] - predictions).hist(bins=20, range=(-10000,10000))
    return plt

def plot_weather_data(data):
    '''
    consume: turnstile_weather
    returns: scatterplot of Average hourly entries by day of the week
    '''

    data['day_of_week'] = pd.to_datetime(data['DATEn']).dt.dayofweek
    # plot = ggplot(data, aes('day_of_week', 'ENTRIESn_hourly')) + geom_point()

    # average of entries ploted against day of week
    table = data.groupby('day_of_week', as_index=False).agg({'ENTRIESn_hourly': np.mean})
    # table['Day of the WEEK'] = table['day_of_week'].apply(str)
    # table.replace(to_replace=['0','1','2','3','4','5','6'], value=['Mon','Tue','Wed','Thur','Fri','Sat','Sun'], inplace=True)

    plot = ggplot(table, aes('day_of_week', 'ENTRIESn_hourly')) +\
    geom_point(color='steelblue', size=200) + xlim(0, 6) + ylim(0, 1500) +\
    ggtitle("Average hourly entries by day of the week") +\
    xlab("Days of the WEEK (0 = Monday, 6 = Sunday)") + ylab("Average hourly ENTRIES")

    return plot

def boxplot_date_time(data_to_plot):
    '''
    consume data made up of different series
    ref: http://blog.bharatbhole.com/creating-boxplots-with-matplotlib/
    '''
    # Create a figure instance
    fig = plt.figure(1, figsize=(9, 6))

    # Create an axes instance
    ax = fig.add_subplot(111)

    # Create the boxplot
    bp = ax.boxplot(data_to_plot)

    # Save the figure
    fig.savefig('fig1.png', bbox_inches='tight')

    ## add patch_artist=True option to ax.boxplot()
    ## to get fill color
    bp = ax.boxplot(data_to_plot, patch_artist=True)

    ## change outline color, fill color and linewidth of the boxes
    for box in bp['boxes']:
        # change outline color
        box.set( color='#7570b3', linewidth=2)
        # change fill color
        box.set( facecolor = '#1b9e77' )

    ## change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='#7570b3', linewidth=2)

    ## change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color='#7570b3', linewidth=2)

    ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='#b2df8a', linewidth=2)

    ## change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='#e7298a', alpha=0.5)

    ## Custom x-axis labels
    ax.set_xticklabels(['Sample1', 'Sample2', 'Sample3', 'Sample4'])

    ## Remove top axes and right axes ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    fig.show()
