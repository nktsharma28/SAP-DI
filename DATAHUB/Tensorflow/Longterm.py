from operators.test_script.function import foo

import pandas as pd
import io
import datetime
import pickle

from functools import partial

import multiprocessing
from multiprocessing import get_context, cpu_count, Pool


def on_input(msg1, msg2):
#def gen():
    
    GROUPBY_LEVEL = ['retailer', 'state', 'brand', 'ppg']
    TARGET = "FillMean"
    DATE_COLUMN = "week_ending_date"
    # Add additional regressors to the forecast
    ADDITIONAL_REGRESSORS = ['food_cpi_nat_mth', 'snap_cost_st_mth']
    REGRESSOR_LAG = {'snap_cost_st_mth': 1}
    
    # Keep track of data throughout run
    data = []
    excluded = []
    df_final_out = pd.DataFrame()
    count=0
    try_count = 0

    # Establish training/test windows
    TRAIN_START = pd.to_datetime('2019-01-01').date()
    TRAIN_END = pd.to_datetime('2020-06-30').date()
    TEST_START = pd.to_datetime('2020-09-30').date()
    TEST_END = pd.to_datetime('2020-12-31').date()
    # To speed up running, set to desired sample size if not set to 0
    SAMPLE = 0
    # Set logistic growth function cap
    CAP_PERCENTILE = 95
    # Future Period
    FUTURE_PERIOD = 25
    # model parameters
    OPTIM_PARAM = {"growth": "logistic", "seasonality_prior_scale": 0.1}
    

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

    
    # Obtain data
    # input table from SEP_COVIDEXT.Z_SEP.AnalyticalModels.SCM.DemandForecasting.CovidExternal::TA_SCM_COVID_FULL_SAMPLE_TRAIN
    # format data frame as per data type in source table
    
    #df_sample_train = pd.read_csv(io.StringIO(msg1.body), sep=",")
    df =  pd.read_json(io.StringIO(msg1))
    #df =  pd.read_csv('/vrep/vflow/scm_demand_sensing_masked_data_imputed_v5.csv')
    df = df[['week_ending_date', 'retailer', 'state', 'business', 'category', 'brand', 'ppg', 'pos_qty_ty', 'pos_dollar_ty', 'FillMean']]
    
    # break according to api.multiplicity and api.multiplicity_index
    batch_size = int(df.shape[0]/api.multiplicity)
    begin_batch = api.multiplicity_index*batch_size
    end_batch = (api.multiplicity_index+1)*batch_size
    
    df = df.iloc[begin_batch:end_batch]
    
    #checking null values and replace accordingly
    df = df.apply(custom_fillna)
    df['week_ending_date'] =  df['week_ending_date'].astype(str)
    
    #checking null values and replace accordingly
    df_add_external =  pd.read_json(io.StringIO(msg2))
    #df_add_external =  pd.read_csv('/vrep/vflow/external_merged_weekly_modeling_source.csv')
    df_add_external = df_add_external[['date', 'state', 'state_initial', 'AT_adj', 'food_cpi_nat_mth', 'snap_cost_st_mth', 'allbed_mean', 'confirmed_infections', 'deaths_mean', 'est_infections_mean', 'mobility_composite_wors', 'states_on_stay_home', 'states_on_travel_limit', 'states_on_any_business', 'states_on_all_non-ess_business', 'states_on_any_gathering_restrict', 'states_on_educational_fac']] 
    
    api.send('output', str(df.shape))
    df_add_external = df_add_external.apply(custom_fillna)
    df_add_external['date'] =  df_add_external['date'].astype(str)
  
    def get_end_date_from_week(year,week,day):

        #Calculates first day and last day of week, given a year and week
        first_day = datetime.datetime.strptime(f'{year}-W{int(week )- 1}-1', '%Y-W%W-%w').date()
        last_day = first_day + datetime.timedelta(days=day)
        return last_day
    
    def get_week_number_from_end_date(date_obj):
        #Calculates week number in year given a date.
        week_number = date_obj.isocalendar()[1]
        return week_number   
    
    def data_imputation(df):
        # Fill in missing week numbers
        df['week_of_year'] = df[DATE_COLUMN].apply(lambda x: get_week_number_from_end_date(x))
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN]).dt.date
        grouped = df.groupby(GROUPBY_LEVEL)
        data1 =  grouped.values.tolist()

        subset_list = []
        for name, group in grouped:
            subset_df = group.copy()
            subset_df = subset_df.set_index(DATE_COLUMN)
            # Imputation approach using mean value
            subset_df = subset_df.assign(imputed_qty=subset_df[TARGET].fillna(subset_df[TARGET].mean()))
            subset_df = subset_df.reset_index()
            subset_list.append(subset_df)
        imputed_df = pd.concat(subset_list)
        #imputed_df.to_csv(DATA_PATH+'processed/imputed_data.csv', index=False)
        return imputed_df

    def add_external_data(df, df_add):
        #Takes client data and external data
        

        df_add = df_add[['date', 'state_initial'] + ADDITIONAL_REGRESSORS]
        df_add.rename(columns={'date': 'week_ending_date', 'state_initial': 'state'}, inplace=True)
        if REGRESSOR_LAG != {}:
            for k, v in REGRESSOR_LAG.items():
                df_add[k] = df_add.groupby('state')[k].shift(v)
        df = pd.merge(df, df_add, on=['week_ending_date', 'state'], how='left')
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
        # Builds a holiday calendar.
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
        df['ds'] = pd.to_datetime(df['ds']).dt.date
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

    # def plot_mape(stats_df):
    #     plt.style.use('ggplot')
    #     first_edge, last_edge = stats_df['mape'].min(), stats_df['mape'].max()

    #     n_equal_bins = 60
    #     bin_edges = np.linspace(start=first_edge, stop=last_edge, num=n_equal_bins + 1, endpoint=True)

    #     # Creating histogram
    #     fig, ax = plt.subplots(figsize =(8, 4))
    #     ax.hist(stats_df['mape'], bins = bin_edges,  color = (0.5,0.1,0.5,0.6))

    #     plt.title('MAPE distribution of forecast results.')

    #     # Save plot
    #     plt.savefig(DATA_PATH+'mape_plot.png')

    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred)/ y_true)) * 100
    

    df, df_add_new = add_external_data(df, df_add_external)
    df_add_new['ds'] =  df_add_new['ds'].astype(str)
    #data =  df.values.tolist()

    # Track how long end-to-end modeling takes
    #load data, date should be in first column
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN]).dt.date
    df[DATE_COLUMN] =  df[DATE_COLUMN].astype(str) #Convert Date to String for DI
    # Load Additional Data
    
    # Get relevant columns
    df = select_data(df)
    #data =  df.values.tolist()

    #get sample, if number provided, else run on full set
    if SAMPLE:
        df = select_sample(df)
    df.rename(columns={DATE_COLUMN: 'ds', TARGET: 'y'}, inplace=True)

    #get Weekday
    week_day = get_week_day(df)
    #create custom holiday
    cust_df_new = custom_holidays(week_day)
    cust_df_new['ds'] =  cust_df_new['ds'].astype(str)
    

    df_add_new['ds'] =  df_add_new['ds'].astype(str)
    
    #train, forecast, and get results
    final_data = []
    excluded_groups = []
    # Rename to prophet's requirements
    df.rename(columns={DATE_COLUMN: 'ds', TARGET: 'y'}, inplace=True)
    # Group data to build individual models at each level
    #grouped= pd.DataFrame()#Force grouped to a dataframe
    #grouped1= pd.DataFrame()
    grouped = df.groupby(GROUPBY_LEVEL) #To check if groupby() is working
    grouped_ak = df.groupby(GROUPBY_LEVEL).count() #by AK
    data = grouped_ak.values.tolist()

    data2 = grouped_ak.values.tolist()

    count=0
    lst=[]
    api.send('output', str(len(grouped.groups)))
    
    pool = multiprocessing.Pool(processes=8)
    with get_context("spawn").Pool() as pool:
        res = pool.map(partial(foo, grouped=grouped, cust_df_new=cust_df_new, df_add_new=df_add_new), list(grouped.groups.keys()), chunksize=435)
        pool.close()
        pool.join()
    api.send('output', 'LINE313')    
    api.send('output', str(res))
    joined_df = pd.concat(res)
    joined_df.reset_index(inplace=True)
    api.send('output', str(joined_df.shape))
    df1 = joined_df[['retailer', 'state', 'brand', 'ppg', 'food_cpi_nat_mth', 'snap_cost_st_mth', 'week_ending_date','FillMean']].copy()
    #api.send("output",str(joined_df.dtypes))
    api.send('output3', df1.to_csv())
    """
    f2 = '{},{},{},{},{},{},{},{},{}'# format Output_Seg
    data=joined_df.values.tolist()
    for j in data:
        api.send("output3",f2.format(*j)+'\n')
    """
    api.send('output2', api.Message(joined_df.to_json()))
        
#api.add_generator(gen)

api.set_port_callback(["input1","input2"], on_input)
    
