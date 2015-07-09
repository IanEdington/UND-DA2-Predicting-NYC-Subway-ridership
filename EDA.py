import analysis as a
import pprint
import matplotlib.pyplot as plt
import numpy as np
### Analyze the binary features ['rain', 'fog', 'thunder']

data = a.read_csv()

# ### What Features could be affecting ridership?
# # all Features:
# print (data.columns.values.tolist())
# #> ['Unnamed: 0', 'UNIT', 'DATEn', 'TIMEn', 'Hour', 'DESCn', 'ENTRIESn_hourly', 'EXITSn_hourly', 'maxpressurei', 'maxdewpti', 'mindewpti', 'minpressurei', 'meandewpti', 'meanpressurei', 'fog', 'rain', 'meanwindspdi', 'mintempi', 'meantempi', 'maxtempi', 'precipi', 'thunder']

# # yes UNIT: using UNIT would tell us if their is a specific area that is being used more than another not weather related we want to predict ridership overall.
# # yes DATEn would tell us if specific days are used more than others. This would be useful if we could seperate days of the week.
# # yes TIMEn would be useful if it was grouped by hours
# yes Hour same as TIMEn
# No DESCn - is only ever REGULAR
# No ENTRIESn_hourly - primary dependent variable
# No EXITSn_hourly - secondary dependent variable
# yes to all weather data
#     'maxpressurei', 'maxdewpti', 'mindewpti', 'minpressurei', 'meandewpti', 'meanpressurei', 'fog', 'rain', 'meanwindspdi', 'mintempi', 'meantempi', 'maxtempi', 'precipi'
# No excet'thunder'
features = ['UNIT', 'DATEn', 'Hour', 'maxpressurei', 'maxdewpti', 'mindewpti', 'minpressurei', 'meandewpti', 'meanpressurei', 'fog', 'rain', 'meanwindspdi', 'mintempi', 'meantempi', 'maxtempi', 'precipi']

### Plot the interesting factors against ENTRIESn_hourly
# plt.scatter(data['ENTRIESn_hourly'], data['rain'])

### bar chart of ENTRIESn_hourly by UNIT
	# http://pandas.pydata.org/pandas-docs/stable/groupby.html
def bar_plot_of_sums(data, feature):
	plt.figure()
	UNIT_data = data[[feature, 'ENTRIESn_hourly']]
	gby_UNIT = UNIT_data.groupby(feature, as_index=False)
	gby_UNIT_sum = gby_UNIT.sum()
	lables = gby_UNIT_sum[feature]
	y_axis = gby_UNIT_sum['ENTRIESn_hourly']
	xlocations = np.array(range(len(y_axis)))+0.5
	width = 0.5
	plt.bar(xlocations, y_axis, width=width)
	plt.show()

for feature in features:
	bar_plot_of_sums(data, feature)


# ### compare binary factors using Mann-Whitney statistic
# bi_f = {
# 	'title' :['mean with_rain', 'mean without_rain', 'Mann-Whitney U-statistic', 'Mann-Whitney p-value' ],
# }
# bi_f['rain'] = list(a.mann_whitney_plus_means(data[data.rain == 0], data[data.rain == 1]))
# bi_f['fog'] = list(a.mann_whitney_plus_means(data[data.fog == 0], data[data.fog == 1]))
# bi_f['thunder'] = list(a.mann_whitney_plus_means(data[data.thunder == 0], data[data.thunder == 1]))
# pprint.pprint (bi_f)
# ### result from 'turnstile_data_master_with_weather.csv'
# { 'title': ['mean with_rain', 'mean without_rain', 'Mann-Whitney U-statistic', 'Mann-Whitney p-value']
#   'fog': [1083.4492820876781, 1154.6593496303688, 1189034717.5, 0.0000060915569104373036],
#   'rain': [1090.278780151855, 1105.4463767458733, 1924409167.0, 0.024999912793489721],
#  'thunder': [1095.3484778440481, nan, 0.0, 0.0],
# }
