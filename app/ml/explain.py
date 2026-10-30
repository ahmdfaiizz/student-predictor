import numpy as np
import shap

class Explainer:
    def __init__(self, pipeline):
        self.prep = pipeline.named_steps["prep"]
        self.model = pipeline.named_steps["model"]
        self.names = self.prep.get_feature_names_out()
        self.explainer = shap.TreeExplainer(self.model)

    def top_factors(self, row_df, k=3):
        X = self.prep.transform(row_df)
        if hasattr(X, "toarray"):
            X = X.toarray()
        values = self.explainer.shap_values(X)
        values = np.asarray(values)[0] if np.asarray(values).ndim > 1 else np.asarray(values)
        order = np.argsort(-np.abs(values))[:k]
        factors = []
        for i in order:
            name = self.names[i].split("__", 1)[-1]
            factors.append({
                "feature": name,
                "impact": round(float(values[i]), 3),
                "direction": "raises risk" if values[i] > 0 else "lowers risk",
            })
        return factors