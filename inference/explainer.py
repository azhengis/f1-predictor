import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shap
import pandas as pd


def get_shap_explanations(model, X: pd.DataFrame) -> list:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    explanations = []
    for i in range(len(X)):
        top = sorted(
            zip(X.columns, shap_values[i]),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]
        explanations.append({feat: round(float(val), 3) for feat, val in top})

    return explanations