import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def evaluate_model(model, test_df, feature_cols):
    results = []
    for race_id, race in test_df.groupby("raceId"):
        preds = model.predict(race[feature_cols])
        predicted_order = race["driverId"].values[np.argsort(preds)]
        actual_order    = race.sort_values("positionOrder")["driverId"].values

        rho, _ = spearmanr(
            np.argsort(predicted_order),
            np.argsort(actual_order)
        )

        top1  = predicted_order[0] == actual_order[0]
        top3  = len(set(predicted_order[:3])  & set(actual_order[:3]))  / 3
        top10 = len(set(predicted_order[:10]) & set(actual_order[:10])) / 10

        results.append({"raceId": race_id, "spearman": rho,
                         "top1": top1, "top3": top3, "top10": top10})

    df = pd.DataFrame(results)
    print("\n=== Evaluation Results ===")
    print(f"Spearman correlation : {df['spearman'].mean():.3f}")
    print(f"Top-1 accuracy       : {df['top1'].mean():.1%}")
    print(f"Top-3 accuracy       : {df['top3'].mean():.1%}")
    print(f"Top-10 accuracy      : {df['top10'].mean():.1%}")
    return df