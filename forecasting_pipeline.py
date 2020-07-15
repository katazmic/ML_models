"""
Pipeline is the class that has the end to end machine learning process. After fitting the data, an instance of this
class can be later used for predictions.
## example use case for this class -- net_deposits forecasts:
from ltv.ml_pipeline.forecasting_pipeline import ForecastingPipeline
import pandas as pd
# for training
pd_full_training_data = pd.read_csv(open(path_to_training))
pd_full_training_data['made_changes'] = 1
pd_full_training_data.loc[abs(pd_full_training_data['actual_net_deposits'] -
                              pd_full_training_data['net_deposits']) <= 0.01, 'made_changes'] = 0
forecasting_pipline = ForecastingPipeline(path_models_params_yaml='model_params_net_deposits.yml',)
forecasting_pipline.fit(pd_full_training_data)
# for predictions
pd_full_prediction_data = pd.read_csv(open(path_to_prediction))
predicted_value = forecasting_pipline.predict(pd_full_prediction_data)
"""

# for a first pass the entire code will be in this class. later iterations might entail splitting up classes
# for pre-processing, training, and predictions

from __future__ import absolute_import

from builtins import object
from itertools import chain

import numpy as np
import yaml
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

DEFAULT_CV_FOLDS = 5
N_TRAINING = 10000

MODELS = {
    'DecisionTreeRegressor': DecisionTreeRegressor,
    'LogisticRegression': LogisticRegression,
    'GradientBoostingRegressor': GradientBoostingRegressor,
    'ElasticNet': ElasticNet
}

COLUMN_NAME = {
    'prediction_range': 'prediction_range',
    'ml_prediction': 'ml_prediction',
    'projection_horizon': 'projection_horizon'
}


class ForecastingPipeline(object):
    def __init__(self, path_models_params_yaml):
        self.classification_models = None
        self.regression_models = None
        self.raw_yaml = yaml.full_load(open(path_models_params_yaml))
        self.models_params = self.raw_yaml['models_params']
        self.pipeline_input = self.raw_yaml['pipeline_input']
        self.features = {'Classification': self.pipeline_input.get('classification_features', []),
                         'Regression': self.pipeline_input.get('regression_features', [])}
        self.target = {'Classification': self.pipeline_input.get('classification_target'),
                       'Regression': self.pipeline_input.get('regression_target')}
        self.forecast_variable = self.pipeline_input.get('forecast_variable')

    def _get_best_model_projection(self, pd_projection_training, modeling_type):
        model_params = self.models_params[modeling_type]
        cv_folds = self.models_params.get('cv_folds', DEFAULT_CV_FOLDS)
        x_train = pd_projection_training[self.features[modeling_type]]
        y_train = pd_projection_training[self.target[modeling_type]]
        best_score = None
        for ml_model_name, hyperparameters in model_params.items():
            ml_model = MODELS[ml_model_name]()
            gs_model = GridSearchCV(ml_model, hyperparameters, cv=cv_folds)
            gs_model.fit(x_train, y_train)
            current_score = gs_model.best_score_
            if not best_score or current_score < best_score:
                best_score = current_score
                best_performing_model = gs_model.best_estimator_
        return best_performing_model

    def fit(self, pd_full_training_data):

        classification_models = {}
        regression_models = {}

        projection_horizons = list(set(pd_full_training_data[COLUMN_NAME['projection_horizon']]))
        for proj_horizon in projection_horizons:
            pd_train_proj = pd_full_training_data[pd_full_training_data[COLUMN_NAME['projection_horizon']]
                                                  == proj_horizon]
            pd_train_proj = pd_train_proj.fillna(pd_train_proj.median())

            if self.target.get("Classification"):
                classification_models[proj_horizon] = self._get_best_model_projection(pd_train_proj, "Classification")
            if self.target.get("Regression"):
                regression_models[proj_horizon] = self._get_best_model_projection(pd_train_proj, "Regression")
        self.classification_models = classification_models
        self.regression_models = regression_models

    def predict(self, pd_full_prediction_data):
        if not self.classification_models and not self.regression_models:
            raise RuntimeError("You need to fit the models before being able to make predictions")

        projection_horizons__prediction = list(set(pd_full_prediction_data[COLUMN_NAME['projection_horizon']]))
        projection_horizons__training = list(self.regression_models.keys()) or list(self.classification_models.keys())

        if len(set(projection_horizons__prediction) - set(projection_horizons__training)) != 0:
            raise ValueError("you cant forecast to projections that are not trained on")
        proba_clf_name = 'probability_{}'.format(self.target['Classification'])
        predicted_value_name = 'predicted_{}'.format(self.forecast_variable)

        for proj_horizon in projection_horizons__prediction:
            pd_pred_proj = pd_full_prediction_data[pd_full_prediction_data[COLUMN_NAME['projection_horizon']]
                                                   == proj_horizon]
            all_features = list(set(chain(*list(self.features.values()))))
            pd_pred_proj = pd_pred_proj[all_features]
            pd_pred_proj = pd_pred_proj.fillna(pd_pred_proj.median())

            clf_mdl = self.classification_models.get(proj_horizon)
            regr_mdl = self.regression_models.get(proj_horizon)

            if clf_mdl:
                index_for_change = np.where(clf_mdl.classes_ == 1)[0][0]
                prob_clf_prj = clf_mdl.predict_proba(pd_pred_proj[self.features['Classification']])[:, index_for_change]
                pd_full_prediction_data.loc[pd_pred_proj.index, proba_clf_name] = prob_clf_prj

            if regr_mdl:
                pred_value_prj = regr_mdl.predict(pd_pred_proj[self.features['Regression']])
                pd_full_prediction_data.loc[pd_pred_proj.index, COLUMN_NAME['ml_prediction']] = pred_value_prj

        prob_clf = pd_full_prediction_data[proba_clf_name] if self.target.get("Classification") else 1.

        if self.target.get("Regression"):
            pred_value = pd_full_prediction_data[COLUMN_NAME['ml_prediction']]
            benchmark_value = pd_full_prediction_data[self.forecast_variable]
        else:
            pred_value = 1
            benchmark_value = 0

        pd_full_prediction_data[predicted_value_name] = (
                pred_value * prob_clf + (1.0 - prob_clf) * benchmark_value
        )

        return pd_full_prediction_data[predicted_value_name]

    def preprocess(self, pd_frame):
        # TODO (katazmic): any pre-processing goes here
        raise NotImplementedError("preprocess is not yet implemented")

    def evaluate(self, pd_holdout):
        # TODO (katazmic)
        raise NotImplementedError("evaluate is not yet implemented")
