#asset_input.py
 
# -*- coding: utf-8 -*-
import os
import sys
from alolib.asset import Asset
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd  
from pathlib import Path
from glob import glob
import numpy as np
import joblib

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBClassifier
import xgboost as xgb

#--------------------------------------------------------------------------------------------------------------------------
#    CLASS
#--------------------------------------------------------------------------------------------------------------------------
class UserAsset(Asset):
    def __init__(self, asset_structure):
        super().__init__(asset_structure)        
        self.args       = self.asset.load_args()
        self.config     = self.asset.load_config()

    @Asset.decorator_run
    def run(self):        
                            
        """ Data load """        

        # ALO API 사용

        # df_path_list = glob('{}/*/*.csv'.format(self.asset.get_input_path()))
        # data = pd.read_csv(df_path_list[-1]) 
        
        data_path = glob('{}/*/*.csv'.format(self.asset.get_input_path()))
        data = pd.read_csv(data_path[-1]) # custom directory 사용
        df = data.dropna()

        
        """ train/validset define """
        # 독립변수/종속변수 정의
        X = df[['HVDC', '풍력(MWh)', '태양광(MWh)', '중유(MWh)', '경유(MWh)', '바이오중유(MWh)', 'demand']]
        y = df['발전제한량']

        # 학습(train)/검증(valid) 데이터셋 정의
        X_train, X_valid, y_train, y_valid = train_test_split(X,
                                                              y,
                                                              test_size=(1-self.args['train_ratio']),
                                                              shuffle=False,
                                                              random_state=self.args['random_state'])
        
        
        """ model define & training """
        # xgboost 모델 정의
        model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

        # xgboost 모델 학습
        model.fit(X_train, y_train)        
        
        
        """ model evaulation """
        # prediction
        y_pred = model.predict(X_valid)

        # evaluation by metrics
        result = {'mse': mean_squared_error(y_valid, y_pred),
                  'mae': mean_absolute_error(y_valid, y_pred),
                  'r2': r2_score(y_valid, y_pred)}

        
        
        """ model save """
        joblib.dump(model, os.path.join(self.asset.get_model_path(), "best_model.joblib"))
        
        
        """ save configuration """
        self.config['data_dir'] = self.args['data_dir']
        self.config['model_file'] = 'best_model.joblib'
        
        
        """ requirement API """
        self.asset.save_data(result) # to next asset
        self.asset.save_config(self.config) # to next asset
                
        
#--------------------------------------------------------------------------------------------------------------------------
#    MAIN
#--------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    ua = UserAsset(envs={}, argv={}, data={}, config={})
    ua.run()
