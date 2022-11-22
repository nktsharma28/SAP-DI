import pandas as pd
import numpy as np
from fbprophet import Prophet


def foo(g, grouped, cust_df_new, df_add_new):
    GROUPBY_LEVEL = ['retailer', 'state', 'brand', 'ppg']
    ADDITIONAL_REGRESSORS = ['food_cpi_nat_mth', 'snap_cost_st_mth']
    REGRESSOR_LAG = {'snap_cost_st_mth': 1}
    

    # Establish training/test windows
    TRAIN_START = pd.to_datetime('2019-01-01').date()
    TRAIN_END = pd.to_datetime('2020-06-30').date()
    TEST_START = pd.to_datetime('2020-09-30').date()
    TEST_END = pd.to_datetime('2020-12-31').date()
    TARGET = "FillMean"
    DATE_COLUMN = "week_ending_date"
    p = {"growth": "logistic", "seasonality_prior_scale": 0.1}
    SAMPLE = 0
    # Set logistic growth function cap
    CAP_PERCENTILE = 95
    # Future Period
    FUTURE_PERIOD = 25
    # model parameters
    OPTIM_PARAM = {"growth": "logistic", "seasonality_prior_scale": 0.1}
    # Keep track of how long it takes to run for 1 group
    
    # Make sure we do not have NaN in additional regressor columns
    df_group = grouped.get_group(g)

    df_group = df_group.sort_values('ds')
    # Set train/predict windows
    train_df = df_group.loc[df_group['ds'] <= TRAIN_END]

    # Set cap for logistic function
    max_val = abs(np.percentile(train_df["y"], CAP_PERCENTILE))
   
    train_df["cap"] = max_val
    

    # Initialize model
    m = Prophet(holidays=cust_df_new, **p)	
    # Add holiday and additional regressors
    m.add_country_holidays(country_name='US')
    
    if ADDITIONAL_REGRESSORS != []:
        for regressor in ADDITIONAL_REGRESSORS:
            m.add_regressor(regressor)


    train_df['ds']= train_df['ds'].astype(str)
    data =  train_df.columns.values.tolist()+train_df.values.tolist()
    
    df_final_out = pd.DataFrame()
    
    # Fit model
    try:
       
        m.fit(train_df)
        # Create future dataframe and predict
        future = m.make_future_dataframe(periods=FUTURE_PERIOD, include_history=False, freq='7D')
        future["cap"] = max_val
        future['ds'] = pd.to_datetime(future['ds']).dt.date		
        
        if ADDITIONAL_REGRESSORS != []:
            future["state"] = g[1]
            df_add_new["ds"] = pd.to_datetime(df_add_new['ds']).dt.date
            future = pd.merge(future, df_add_new, on=['ds', 'state'], how='left')	


        future = future.dropna()
        forecast = m.predict(future)
        forecast['ds'] = pd.to_datetime(forecast['ds']).dt.date	
        df_final = forecast.loc[(forecast['ds'] >= TEST_START) & (forecast['ds'] <= TEST_END)]
        df_final = df_final[['ds', 'yhat'] + ADDITIONAL_REGRESSORS]
        df_final.rename(columns={'ds': DATE_COLUMN, 'yhat': TARGET}, inplace=True)
        for i in range(len(GROUPBY_LEVEL)):
            df_final[GROUPBY_LEVEL[i]] = g[i]
        df_final = df_final[GROUPBY_LEVEL + ADDITIONAL_REGRESSORS + [DATE_COLUMN, TARGET]]	
        df_final_out = df_final_out.append(df_final)
        data =  df_final.values.tolist()
        #del forecast, df_final, future, m, df_group, train_df
    except Exception as e:
        #excluded.append(g[:len(GROUPBY_LEVEL)])
        df_final = pd.DataFrame()
        data =  df_final.values.tolist()
        #del df_final,  m, df_group, train_df
    return df_final_out