from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from config import METRICS_DIR, create_output_directories


def main() -> None:
    create_output_directories()

    path_1x1 = (
        METRICS_DIR
        / "convlstm_many_centers_9892_by_point_metrics.csv"
    )

    path_5x5 = (
        METRICS_DIR
        / "convlstm_all_points_masked_5x5_by_point_metrics.csv"
    )

    df_1x1 = pd.read_csv(path_1x1)
    df_5x5 = pd.read_csv(path_5x5)

    keep_cols = [
        "center_index",
        "LAMBX",
        "LAMBY",
        "persistence_MAE_C",
        "persistence_RMSE_C",
        "convlstm_MAE_C",
        "convlstm_RMSE_C",
        "convlstm_BIAS_C",
        "convlstm_R2",
        "MAE_improvement_percent",
        "RMSE_improvement_percent",
    ]

    df_1x1 = df_1x1[keep_cols].copy()
    df_5x5 = df_5x5[keep_cols + ["existing_cells_5x5", "complete_5x5"]].copy()

    df_1x1 = df_1x1.rename(
        columns={
            "convlstm_MAE_C": "MAE_1x1",
            "convlstm_RMSE_C": "RMSE_1x1",
            "convlstm_BIAS_C": "BIAS_1x1",
            "convlstm_R2": "R2_1x1",
            "MAE_improvement_percent": "MAE_gain_1x1",
            "RMSE_improvement_percent": "RMSE_gain_1x1",
        }
    )

    df_5x5 = df_5x5.rename(
        columns={
            "convlstm_MAE_C": "MAE_5x5",
            "convlstm_RMSE_C": "RMSE_5x5",
            "convlstm_BIAS_C": "BIAS_5x5",
            "convlstm_R2": "R2_5x5",
            "MAE_improvement_percent": "MAE_gain_5x5",
            "RMSE_improvement_percent": "RMSE_gain_5x5",
        }
    )

    merged = df_1x1.merge(
        df_5x5[
            [
                "center_index",
                "MAE_5x5",
                "RMSE_5x5",
                "BIAS_5x5",
                "R2_5x5",
                "MAE_gain_5x5",
                "RMSE_gain_5x5",
                "existing_cells_5x5",
                "complete_5x5",
            ]
        ],
        on="center_index",
        how="inner",
    )

    merged["MAE_delta_5x5_minus_1x1"] = (
        merged["MAE_5x5"] - merged["MAE_1x1"]
    )

    merged["RMSE_delta_5x5_minus_1x1"] = (
        merged["RMSE_5x5"] - merged["RMSE_1x1"]
    )

    merged["five_x_five_better_MAE"] = (
        merged["MAE_delta_5x5_minus_1x1"] < 0
    )

    merged["five_x_five_better_RMSE"] = (
        merged["RMSE_delta_5x5_minus_1x1"] < 0
    )

    merged["not_improved_1x1"] = merged["MAE_gain_1x1"] <= 0
    merged["not_improved_5x5"] = merged["MAE_gain_5x5"] <= 0

    output_detail = (
        METRICS_DIR
        / "comparison_1x1_vs_masked_5x5_by_point.csv"
    )

    merged.to_csv(output_detail, index=False)

    old_bad = merged[merged["not_improved_1x1"]].copy()
    new_bad = merged[merged["not_improved_5x5"]].copy()

    corrected_old_bad = old_bad[~old_bad["not_improved_5x5"]].copy()
    still_bad = old_bad[old_bad["not_improved_5x5"]].copy()
    newly_bad = merged[
        (~merged["not_improved_1x1"])
        & (merged["not_improved_5x5"])
    ].copy()

    summary = {
        "number_of_points": int(len(merged)),
        "MAE_1x1_mean": float(merged["MAE_1x1"].mean()),
        "MAE_5x5_mean": float(merged["MAE_5x5"].mean()),
        "RMSE_1x1_mean": float(merged["RMSE_1x1"].mean()),
        "RMSE_5x5_mean": float(merged["RMSE_5x5"].mean()),
        "points_where_5x5_better_MAE": int(merged["five_x_five_better_MAE"].sum()),
        "points_where_1x1_better_MAE": int((~merged["five_x_five_better_MAE"]).sum()),
        "points_where_5x5_better_RMSE": int(merged["five_x_five_better_RMSE"].sum()),
        "points_where_1x1_better_RMSE": int((~merged["five_x_five_better_RMSE"]).sum()),
        "not_improved_1x1": int(merged["not_improved_1x1"].sum()),
        "not_improved_5x5": int(merged["not_improved_5x5"].sum()),
        "old_15_corrected_by_5x5": int(len(corrected_old_bad)),
        "old_15_still_not_improved": int(len(still_bad)),
        "new_points_not_improved_only_in_5x5": int(len(newly_bad)),
    }

    output_summary = (
        METRICS_DIR
        / "comparison_1x1_vs_masked_5x5_summary.json"
    )

    with output_summary.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=4)

    corrected_old_bad.to_csv(
        METRICS_DIR / "old_1x1_bad_points_corrected_by_5x5.csv",
        index=False,
    )

    still_bad.to_csv(
        METRICS_DIR / "old_1x1_bad_points_still_bad_in_5x5.csv",
        index=False,
    )

    newly_bad.to_csv(
        METRICS_DIR / "new_bad_points_only_in_5x5.csv",
        index=False,
    )

    print("=" * 80)
    print("COMPARAISON 1×1 VS 5×5 MASQUÉ")
    print("=" * 80)
    print(json.dumps(summary, ensure_ascii=False, indent=4))

    print("\nAnciens 15 points corrigés par le 5×5 :")
    if len(corrected_old_bad) == 0:
        print("Aucun")
    else:
        print(
            corrected_old_bad[
                [
                    "center_index",
                    "LAMBX",
                    "LAMBY",
                    "MAE_1x1",
                    "MAE_5x5",
                    "MAE_gain_1x1",
                    "MAE_gain_5x5",
                    "MAE_delta_5x5_minus_1x1",
                    "existing_cells_5x5",
                ]
            ].to_string(index=False)
        )

    print("\nAnciens 15 points encore non améliorés en 5×5 :")
    if len(still_bad) == 0:
        print("Aucun")
    else:
        print(
            still_bad[
                [
                    "center_index",
                    "LAMBX",
                    "LAMBY",
                    "MAE_1x1",
                    "MAE_5x5",
                    "MAE_gain_1x1",
                    "MAE_gain_5x5",
                    "MAE_delta_5x5_minus_1x1",
                    "existing_cells_5x5",
                ]
            ].to_string(index=False)
        )

    print("\nNouveaux points non améliorés uniquement en 5×5 :")
    print(
        newly_bad[
            [
                "center_index",
                "LAMBX",
                "LAMBY",
                "MAE_1x1",
                "MAE_5x5",
                "MAE_gain_1x1",
                "MAE_gain_5x5",
                "MAE_delta_5x5_minus_1x1",
                "existing_cells_5x5",
            ]
        ]
        .sort_values("MAE_gain_5x5")
        .head(30)
        .to_string(index=False)
    )

    print("\nFichiers créés :")
    print(f"- Comparaison complète : {output_detail}")
    print(f"- Résumé               : {output_summary}")


if __name__ == "__main__":
    main()
