def on_input(msg1, msg2, msg3, msg4):

	#import Libraries
	import io
	import importlib
	import pandas as pd
	import os
	import us
	from datetime import datetime
	import datetime
	import pandas as pd
	import numpy as np
	import ast
	import time
	from pathlib import Path
	from fbprophet import Prophet
	from datetime import date
	from pathlib import Path
	from itertools import repeat
	import matplotlib.pyplot as plt
	import hana_ml.dataframe as dataframe
	#from notebook_hana_connector.notebook_hana_connector import NotebookConnectionContext	
	
	api.send("output2", 'Line 24 After Libraries')
	#variables and constant
	# Set forecast target and choose date column name
	TARGET = 'FillMean'
	DATE_COLUMN = 'week_ending_date'

	# Establish training/test windows
	TRAIN_START = pd.to_datetime('2020-01-01').date()
	TRAIN_END = pd.to_datetime('2020-06-30').date()

	#Test Window is not applicable for Short Term Forecasting
	TEST_START = pd.to_datetime('2020-07-01').date()
	TEST_END = pd.to_datetime('2020-09-30').date() 

	# Choose model features and forecast groupby level
	#INTERNAL_DATA_FILE = "sample_packsizes_v1.csv"
	#INTERNAL_DATA_FILE = "scm_demand_sensing_masked_data_scenario_2.csv"
	#EXTERNAL_DATA_FILE = "usa_risk_index.csv"
	ADDITIONAL_REGRESSORS = ['mobility_index','yearly_scaled','risk_index']
	REGRESSOR_LAG = {'mobility_index':4}
	GROUPBY_LEVEL = ['retailer','state','brand','ppg']

	# To speed up running, set to true and designate what number of samples to run on
	SAMPLE = 10
	#SAMPLE_SIZE = 10
	# Set logistic growth function cap
	CAP_PERCENTILE = 95
	# Future Period
	FUTURE_PERIOD = 25
	#MAX_MODEL_ITER = 50
	# model parameters
	OPTIM_PARAM = {"growth": "logistic", "seasonality_prior_scale": 0.1}
	data = []
	excluded = []
	df_final_out = pd.DataFrame()
	api.send("output2", 'Line 58 Before Functions definition')
	#functions
	def get_end_date_from_week(year,week,day):
		
		"""Calculates first day and last day of week, given a year and week."""
		first_day = datetime.datetime.strptime(f'{year}-W{int(week )- 1}-1', '%Y-W%W-%w').date()
		last_day = first_day + datetime.timedelta(days=day)
		return last_day

	def get_week_number_from_end_date(date_obj):
		"""Calculates week number in year given a date."""
		week_number = date_obj.isocalendar()[1]
		return week_number


	def data_imputation(df):
		# Fill in missing week numbers
		df['week_of_year'] = df[DATE_COLUMN].apply(lambda x: get_week_number_from_end_date(x))
		df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN]).dt.date
		grouped = df.groupby(GROUPBY_LEVEL)
		subset_list = []
		for name, group in grouped:
			subset_df = group.copy()
			subset_df = subset_df.set_index(DATE_COLUMN)
			# Imputation approach using mean value
			subset_df = subset_df.assign(imputed_qty=subset_df[TARGET].fillna(subset_df[TARGET].mean()))
			subset_df = subset_df.reset_index()
			subset_list.append(subset_df)
		imputed_df = pd.concat(subset_list)
		imputed_df.to_csv(DATA_PATH+'processed/imputed_data.csv', index=False)
		return imputed_df


	def add_external_data(df, df_add):
		""" Takes client data and adds external data
		"""

		# # Load additional external data
		#df_add = pd.read_csv(DATA_PATH + EXTERNAL_DATA_FILE)
		df_add = df_add[['date', 'state_initial'] + ADDITIONAL_REGRESSORS]
		df_add.rename(columns={'date': 'week_ending_date', 'state_initial': 'state'}, inplace=True)
		if REGRESSOR_LAG != {}:
			for k, v in REGRESSOR_LAG.items():
				df_add[k] = df_add.groupby('state')[k].shift(v)
					
		#print(df.dtypes)
		#print(df_add.dtypes)
		df_add['week_ending_date'] =  df_add['week_ending_date'].astype(str)
		df = pd.merge(df, df_add, on=['week_ending_date', 'state'], how='left')
		df['week_ending_date'] =  df['week_ending_date'].astype(str)
		df_add[DATE_COLUMN] = pd.to_datetime(df_add[DATE_COLUMN]).dt.date
		df_add.rename(columns={DATE_COLUMN: 'ds'}, inplace=True)
		return df, df_add


	def select_data(df):
		if ADDITIONAL_REGRESSORS:
			df = df[[DATE_COLUMN] + GROUPBY_LEVEL + ADDITIONAL_REGRESSORS + [TARGET]]
		else:
			df = df[[DATE_COLUMN] + GROUPBY_LEVEL + [TARGET]]
		print('Selected data columns.')
		
		return df

	def select_sample(df):
		df_sum = df.groupby(GROUPBY_LEVEL)[[TARGET]].sum().reset_index().rename(columns={TARGET: 'total_qty'})
		top_100 = df_sum.nlargest(columns='total_qty', n=SAMPLE)
		top_100 = top_100.drop(['total_qty'], axis=1)
		df = pd.merge(top_100, df, how='inner', on=GROUPBY_LEVEL)
		print('Chose top {} samples.'.format(SAMPLE))
		return df

	def load_holiday_calendar():
		""" Builds a holiday calendar."""
		# New year's day
		newyear = pd.DataFrame({
		'holiday': 'newyear',
		'ds': pd.to_datetime(['2019-01-01','2020-01-01']),
		})
		# Martin Luther King Jr. Day
		MLK_day = pd.DataFrame({
		'holiday': 'MLK_day',
		'ds': pd.to_datetime(['2019-01-21','2020-01-20']),
		})
		# March Madness
		march_madness = pd.DataFrame({
		'holiday': 'march_madness',
		'ds': pd.to_datetime(['2018-03-24','2018-03-31','2019-03-30','2019-04-06','2020-03-28','2020-04-04', '2021-03-27','2021-04-03']),
		})
		# Superbowl
		superbowls = pd.DataFrame({
		'holiday': 'superbowl',
		'ds': pd.to_datetime(['2018-01-27','2018-02-03', '2019-01-26','2019-02-02', '2020-01-25','2020-02-01','2021-01-30','2021-02-06']),
		})
		# Lent
		lent = pd.DataFrame({
		'holiday': 'lent',
		'ds': pd.to_datetime(['2018-02-17', '2018-02-24','2018-03-03','2018-03-10','2018-03-17','2018-03-24',
							'2019-03-09', '2019-03-16','2019-03-23','2019-03-30','2019-04-06','2019-04-13',
							'2020-02-29', '2020-03-07', '2020-03-14','2020-03-21','2020-03-28','2020-04-04',
							'2021-02-20', '2021-02-27', '2021-03-06','2021-03-13','2021-03-20','2021-03-27']),
		})
		# Easter (Wednesday – Easter Friday)
		easter = pd.DataFrame({
		'holiday': 'easter',
		'ds': pd.to_datetime(['2018-03-31', '2019-04-20', '2020-04-11','2021-04-03']),
		})
		# Memorial day
		memorial_day = pd.DataFrame({
		'holiday': 'memorial_day',
		'ds': pd.to_datetime(['2019-05-27', '2020-05-25']),
		})
		# Independence day
		indep_day = pd.DataFrame({
		'holiday': 'indep_day',
		'ds': pd.to_datetime(['2019-07-04', '2020-07-03']),
		})
		# Labor day
		labor_day = pd.DataFrame({
		'holiday': 'indep_day',
		'ds': pd.to_datetime(['2019-09-02', '2020-09-07']),
		})
		# Halloween
		halloween = pd.DataFrame({
		'holiday': 'halloween',
		'ds': pd.to_datetime(['2018-10-27', '2019-10-26', '2020-10-31','2021-10-30']),
		})
		# Veteran's day
		veteran_day = pd.DataFrame({
		'holiday': 'veteran_day',
		'ds': pd.to_datetime(['2019-11-11', '2020-11-11']),
		})
		# Thanksgiving
		thanksgiving = pd.DataFrame({
		'holiday': 'thanksgiving',
		'ds': pd.to_datetime(['2019-11-28', '2020-11-26']),
		})
		# Christmas
		Christmas = pd.DataFrame({
		'holiday': 'thanksgiving',
		'ds': pd.to_datetime(['2019-12-25', '2020-12-25']),
		})

		holidays_df = pd.concat((newyear, MLK_day, march_madness, superbowls, lent, easter, memorial_day, indep_day, labor_day, halloween, veteran_day, thanksgiving, Christmas))
		return holidays_df

	def get_week_day(df):
		return df['ds'].iloc[0].weekday()

	def custom_holidays(week_day):
		custom_holidays = load_holiday_calendar()
		custom_holidays['week_no'] = custom_holidays['ds'].apply(lambda x: get_week_number_from_end_date(x))
		custom_holidays['year'] = custom_holidays['ds'].apply(lambda x: int(x.strftime('%Y')))
		custom_holidays['week_ending_date'] = custom_holidays.apply(
			lambda x: get_end_date_from_week(x['year'], x['week_no'], week_day), 1)
		custom_holidays.rename(columns={'ds': 'date', 'week_ending_date': 'ds'}, inplace=True)
		custom_holidays = custom_holidays[['ds', 'holiday']]

		return custom_holidays

	def plot_mape(stats_df):
		plt.style.use('ggplot')
		first_edge, last_edge = stats_df['mape'].min(), stats_df['mape'].max()

		n_equal_bins = 60
		bin_edges = np.linspace(start=first_edge, stop=last_edge, num=n_equal_bins + 1, endpoint=True)

		# Creating histogram
		fig, ax = plt.subplots(figsize =(8, 4))
		ax.hist(stats_df['mape'], bins = bin_edges,  color = (0.5,0.1,0.5,0.6))

		plt.title('MAPE distribution of forecast results.')

		# Save plot
		plt.savefig(DATA_PATH+'mape_plot.png')

	def mean_absolute_percentage_error(y_true, y_pred):
		y_true, y_pred = np.array(y_true), np.array(y_pred)
		return np.mean(np.abs((y_true - y_pred)/ y_true)) * 100

	def custom_fillna(series):
		if series.dtype is pd.np.dtype(float):
			return series.fillna(0)
		elif series.dtype is pd.np.dtype('int32'):
			return series.fillna(0)
		elif series.dtype is pd.np.dtype('int64'):
			return series.fillna(0)
		elif series.dtype is pd.np.dtype(str):
			return series.fillna(0)    
		elif series.dtype is pd.np.dtype(object):
			return series.fillna('')
		else:
			return series    
		
	def check_NaN(df):
		try:
			df_nan = df.isna()
			nan_columns = df_nan.any()
			columns_with_nan = df_nan.columns[nan_columns].tolist()
		except Exception as e:
			print(e)
		return columns_with_nan   	
	
	# Load historical data
	df_hist_sample = pd.read_json(io.StringIO(msg1))
	#df_hist_sample['week_ending_date'] =  df_hist_sample['week_ending_date'].astype(str)
	df_hist_sample = df_hist_sample.apply(custom_fillna)
	
	# Load Mobility data
	df_add = pd.read_json(io.StringIO(msg4))
	
	#df_add['week_ending_date'] = pd.to_datetime(df_add['week_ending_date'])
	#df_add['week_ending_date'] =  df_add['week_ending_date'].astype('datetime64[ns]')
	df_add = df_add.apply(custom_fillna)	
	#if column name is in CAPS
	df_add = df_add.rename(columns={'WEEK_ENDING_DATE':'week_ending_date', 'STATE':'state', 'MOBILITY_INDEX':'mobility_index'})	
	df_add[DATE_COLUMN] = pd.to_datetime(df_add[DATE_COLUMN]).dt.date
	df_add[DATE_COLUMN]=df_add[DATE_COLUMN].apply(lambda x: x.strftime('%Y-%m-%d'))	
	
	# Load Seasonality data 	
	df_season =  pd.read_json(io.StringIO(msg2))
	df_season = df_season.apply(custom_fillna)
	#df_season['week_ending_date'] =  df_season['week_ending_date'].astype(str)
	df_season[DATE_COLUMN] = pd.to_datetime(df_season[DATE_COLUMN]).dt.date
	df_season[DATE_COLUMN]=df_season[DATE_COLUMN].apply(lambda x: x.strftime('%Y-%m-%d'))	
	
	
	
	#Load Covid Risk Index data 
	df_external = pd.read_json(io.StringIO(msg3))
	#df_external['week_ending_date'] =  df_external['week_ending_date'].astype(str)
	df_external = df_external.apply(custom_fillna)
	df_external = df_external.rename(columns={'STATE':'state','DATE':'date','CASES':'cases','FORECAST':'forecast','LOG_CASES':'log_cases',
	    'LOG_FORECAST':'log_forecast','RISK_INDEX':'risk_index'})	
	
	#Process Covid Risk Index data Aggreate covid risk index to state level, using formula:
	#Index_state = log(cases_state) + a*log(VIX), where a = corr(log(cases_state), log(VIX))

	# Get sum of all cases within state
	df_covid_state = df_external.groupby(['state','date'])['cases'].sum().reset_index()

	# Get vix forecast by state, same for each county, so simply take the mean
	vix = df_external.groupby(['state','date'])['forecast'].mean().reset_index()

	# Merge cases and vix together
	df_covid_state = df_covid_state.merge(vix, how='left', on=['state', 'date'])

	# Add columns for the log transforms, and fill in inf with 0.0
	df_covid_state['log_cases'] = df_covid_state['cases'].apply(np.log)
	df_covid_state['log_forecast'] = df_covid_state['forecast'].apply(np.log)
	df_covid_state = df_covid_state.replace([np.inf, -np.inf], 0.0)

	# Calculate 'a' factor by calculating correlation between cases and VIX forecast
	a_factor = df_covid_state.groupby('state')[['log_cases','log_forecast']].corr().iloc[0::2,-1].reset_index().drop('level_1', axis=1)
	a_factor.rename({'log_forecast':'a_factor'}, inplace=True, axis=1)

	# Add 'a' factor column to covid state dataframe
	df_covid_state = df_covid_state.merge(a_factor, how='left', on='state')

	# Calculate covid risk index
	df_covid_state['risk_index'] = df_covid_state['log_cases'] + df_covid_state['a_factor']*df_covid_state['log_forecast']

	# Add state abbreviations to dataframe
	df_covid_state['state_abbr'] = df_covid_state['state'].apply(lambda x: us.states.lookup(x).abbr)
	df_covid_state.drop(['state'], axis=1, inplace=True)

	# Clean up date column and rename columns
	df_covid_state['date'] = pd.to_datetime(df_covid_state['date']).dt.date
	df_covid_state = df_covid_state.rename(columns={'date':'week_ending_date', 'state_abbr':'state'})	
	
	df_covid_state = df_covid_state.dropna()
	df_covid_state = df_covid_state.apply(custom_fillna)
	
	df_covid_state[DATE_COLUMN] = pd.to_datetime(df_covid_state[DATE_COLUMN]).dt.date
	df_covid_state[DATE_COLUMN]=df_covid_state[DATE_COLUMN].apply(lambda x: x.strftime('%Y-%m-%d'))
	
	api.send("output2", 'Line 323 Before Processing section')
	
	#Processing Mobility data
	if REGRESSOR_LAG != {}:
		for k, v in REGRESSOR_LAG.items():
			df_add[k] = df_add.groupby('state')[k].shift(v)
	
	
	#merge historical data with mobility external data
	api.send("output2", 'Line 330 Before Merge')
	df1 = pd.merge(df_hist_sample, df_add, on = ['week_ending_date', 'state'], how = 'left').fillna(1)# dfhist changed to df_hist_sample
	#merge resultant data frame with seasonality data
	df2 = pd.merge(df1, df_season, on = ['week_ending_date','retailer','state','brand','ppg'], how = 'left')\
	.fillna(df_season['yearly_scaled'].mean(),axis=0)
	#merge resultant data with covid data
	df_final = pd.merge(df2, df_covid_state, on=['week_ending_date', 'state'], how='left').fillna(0)	
	
	df=df_final.copy()
	
	# Make sure week ending date is set to date and rename column to prophet
	df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN]).dt.date
	
	# Get relevant columns
	df = select_data(df)
	# Get sample, if number provided, else run on full set
	if SAMPLE:
		df = select_sample(df)
	# Rename to prophet's requirements    
	df.rename(columns={DATE_COLUMN: 'ds', TARGET: 'y'}, inplace=True)
	# Get Weekday
	week_day = get_week_day(df)
	# Create custom holiday
	cust_df_new = custom_holidays(week_day)	

	#groupby with pre-defined list of values
	grouped = df.groupby(GROUPBY_LEVEL)	

	#check unique combinations
	for g in grouped.groups:
		api.send("output2",g)	
		
	#single thread implementation
	# Keep track of how long it takes to run for all groups

	for g in grouped.groups:
	    api.send("output2", 'Line 365 Inside For Loop')
	    p = OPTIM_PARAM.copy()
	    for i in range(len(GROUPBY_LEVEL)):
	        data.append(g[i])
	    df_group = grouped.get_group(g)
	    df_group['ds'] = pd.to_datetime(df_group['ds']).dt.date
	    df_group = df_group.sort_values('ds')
	    # Set train/predict windows
	    train_df = df_group.loc[df_group['ds'] <= TRAIN_END]
	    # Make sure we do not have NaN in additional regressor columns
	    max_val = abs(np.percentile(train_df["y"], CAP_PERCENTILE))
	    #min_val = abs(np.percentile(train_df["y"], 5))
	    train_df["cap"] = max_val
	    # train_df["floor"] = min_val
	    # Initialize model
	    m = Prophet(holidays=cust_df_new, **p)
	    # Add holiday and additional regressors
	    m.add_country_holidays(country_name='US')
	    # Add additional regressors
	    if ADDITIONAL_REGRESSORS != []:
	        for regressor in ADDITIONAL_REGRESSORS:
	            m.add_regressor(regressor)
	        api.send("output2", 'Line 396 After Processing')
	        # Fit model
	    try:
	        m.fit(train_df)
	        api.send("output2", 'Line 400 Inside Fit Model')
	        # Create future dataframe and predict
	        future = m.make_future_dataframe(periods=FUTURE_PERIOD, include_history=False, freq='7D')
	        future["cap"] = max_val
	        future['ds'] = pd.to_datetime(future['ds']).dt.date
	        if ADDITIONAL_REGRESSORS != []:
	            future["state"] = g[1]
	            df["ds"] = pd.to_datetime(df['ds']).dt.date
	            df['ds']=df['ds'].apply(lambda x: x.strftime('%Y-%m-%d'))
	            future['ds'] = future['ds'].apply(lambda x: x.strftime('%Y-%m-%d'))
	            future = pd.merge(future, df, on=['ds', 'state'], how='left')
	        #process future dataframe NaN values
	        future['mobility_index'].fillna(1, inplace=True)
	        future['yearly_scaled'].fillna(future['yearly_scaled'].mean(),axis=0, inplace=True) 
	        future['risk_index'].fillna(future['yearly_scaled'].mean(),axis=0, inplace=True)
	        #process future dropduplicate records based on 'ds', if anything entered due to erroneous merge operation 
	        future = future.drop_duplicates(subset='ds', keep="first").reset_index(drop=True)
	        forecast = m.predict(future)
	        forecast['ds'] = pd.to_datetime(forecast['ds']).dt.date
	        df_final = forecast.loc[(forecast['ds'] >= TEST_START) & (forecast['ds'] <= TEST_END)]
	        df_final = df_final[['ds', 'yhat'] + ADDITIONAL_REGRESSORS]
	        df_final.rename(columns={'ds': DATE_COLUMN, 'yhat': TARGET}, inplace=True)
	        for i in range(len(GROUPBY_LEVEL)):
	            api.send("output2", 'Line 429 Inside For Loop')
	            df_final[GROUPBY_LEVEL[i]] = g[i]
	        df_final = df_final[GROUPBY_LEVEL + ADDITIONAL_REGRESSORS + [DATE_COLUMN, TARGET]]
	        df_final_out = df_final_out.append(df_final).reset_index(drop=True)
	        del forecast, df_final, future, m, df_group, train_df
	    except Exception as e:
	        #print(e)
	        excluded.append(g[:len(GROUPBY_LEVEL)])
	        df_final = pd.DataFrame()
	        del df_final, m, df_group, train_df
	#df_final_out will be captured into Hana Table	
	#df_final['week_ending_date'] =  df_final['week_ending_date'].astype(str)
	data = df_final.values.tolist()
	api.send("output2", data)
	
api.set_port_callback(["input1","input2", "input3", "input4"], on_input)