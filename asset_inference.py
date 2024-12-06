#asset_[step_name].py

# -*- coding: utf-8 -*-
import os
import sys
from alolib.asset import Asset
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd

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
from glob import glob
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

        """ data load """

        # ALO API 사용
        df_path_list = glob('{}/*/*.csv'.format(self.asset.get_input_path()))
        data = pd.read_csv(df_path_list[-1])
        # data = pd.read_csv(self.asset.get_input_path()) # custom directory 사용

        X_infer = data[['HVDC', '풍력(MWh)', '태양광(MWh)', '중유(MWh)', '경유(MWh)', '바이오중유(MWh)', 'demand']]

        """ model load """
        model = joblib.load(os.path.join(self.asset.get_model_path(), self.args['model_file']))


        """ model inference """
        prediction = model.predict(X_infer)

        data['pred_class'] = prediction


        """ result save """
        output_path = self.asset.get_output_path() # needed: .csv only 1 / .jpg only 1 / .only, .jpg each 1
        data.to_csv(output_path + 'output.csv')

        summary = {}
        summary['result'] = 'OK'
        summary['score'] = 0.98
        summary['note'] = "The score represents the probability value of the model's prediction result."
        self.asset.save_summary(result=summary['result'], score=summary['score'], note=summary['note'])


        """ requirement API """
        self.asset.save_data(self.args)
        self.asset.save_config(self.config)



#--------------------------------------------------------------------------------------------------------------------------
#    MAIN
#--------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    envs, argv, data, config = {}, {}, {}, {}
    ua = UserAsset(envs, argv, data, config)
    ua.run()
