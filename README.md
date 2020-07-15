# ML_models

forecasting_pipeline = ForecastingPipeline(params.yaml)
logging.info("reading file.")
pd_training_features = pd.read_csv(features.name, engine='python')
logging.info("fitting model")
forecasting_pipeline.fit(pd_training_features)
