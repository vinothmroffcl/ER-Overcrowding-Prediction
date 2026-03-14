from services.ml_service import ml_service

print("RF Features:", ml_service.rf_model.feature_names_in_)
print("XGB Features:", ml_service.xgb_model.feature_names_in_)
