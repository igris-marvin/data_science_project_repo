# ^ IMPORTS
import warnings

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

warnings.filterwarnings("ignore")

# Reproducible Pipeline

# Sequence: Feature Engineer → Imputer → Scaler → SMOTE → Model
# Custom Leakage-Safe Feature Engineering Transformer
class WSAFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Engineers domain-specific water metrics inside the pipeline
    using raw population and access counts retained in X.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = pd.DataFrame(X).copy()

        # Raw population and access counts
        pop = X_out["Population"]
        pop_access = X_out["Population with access to water"]
        urban_pop = X_out["Urban Population"]
        urban_access = X_out["Urban Population with access to water"]
        rural_pop = X_out["Rural Population"]
        rural_access = X_out["Rural Population with access to water"]

        # Engineered features
        X_out["Urban_Ratio"] = (urban_pop / (pop + 1e-5)) * 100
        X_out["Water_Access_Ratio"] = (pop_access / (pop + 1e-5)) * 100
        X_out["Unserved_Population"] = pop - pop_access

        # Urban vs Rural access disparity gap
        urban_acc_pct = (urban_access / (urban_pop + 1e-5)) * 100
        rural_acc_pct = (rural_access / (rural_pop + 1e-5)) * 100
        X_out["Urban_Rural_Access_Gap"] = urban_acc_pct - rural_acc_pct

        return X_out
