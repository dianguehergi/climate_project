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


def load_json(path: Path):
    if not path.is_file():
        print(f"Fichier absent : {path}")
        return None

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def add_row(
    rows,
    experiment,
    model,
    scale,
    patch,
    metrics,
    persistence_metrics=None,
    extra=None,
):
    row = {
        "experiment": experiment,
        "model": model,
        "scale": scale,
        "patch": patch,
        "MAE_C": metrics.get("MAE_C"),
        "RMSE_C": metrics.get("RMSE_C"),
        "BIAS_C": metrics.get("BIAS_C"),
        "R2": metrics.get("R2"),
    }

    if persistence_metrics is not None:
        row["persistence_MAE_C"] = persistence_metrics.get("MAE_C")
        row["persistence_RMSE_C"] = persistence_metrics.get("RMSE_C")

        if metrics.get("MAE_C") is not None:
            row["MAE_gain_vs_persistence_percent"] = (
                100
                * (
                    persistence_metrics["MAE_C"]
                    - metrics["MAE_C"]
                )
                / persistence_metrics["MAE_C"]
            )

        if metrics.get("RMSE_C") is not None:
            row["RMSE_gain_vs_persistence_percent"] = (
                100
                * (
                    persistence_metrics["RMSE_C"]
                    - metrics["RMSE_C"]
                )
                / persistence_metrics["RMSE_C"]
            )

    if extra:
        row.update(extra)

    rows.append(row)


def main() -> None:
    create_output_directories()

    rows = []

    # 100 centres 5x5
    metrics_100 = load_json(
        METRICS_DIR / "convlstm_many_centers_100_metrics.json"
    )

    if metrics_100:
        add_row(
            rows,
            experiment="ConvLSTM 100 centres",
            model="ConvLSTM",
            scale="100 centres SAFRAN",
            patch="5x5 complet",
            metrics=metrics_100["convlstm_metrics"],
            persistence_metrics=metrics_100["persistence_metrics"],
            extra={
                "number_of_points": metrics_100["number_of_centers"],
                "parameters": metrics_100["number_of_parameters"],
                "best_epoch": metrics_100["best_epoch"],
            },
        )

    # Tous points 1x1
    metrics_9892_1x1 = load_json(
        METRICS_DIR / "convlstm_many_centers_9892_metrics.json"
    )

    summary_9892_1x1 = load_json(
        METRICS_DIR / "convlstm_many_centers_9892_by_point_summary.json"
    )

    if metrics_9892_1x1:
        add_row(
            rows,
            experiment="ConvLSTM tous points 1x1",
            model="ConvLSTM",
            scale="9 892 points SAFRAN",
            patch="1x1",
            metrics=metrics_9892_1x1["convlstm_metrics"],
            persistence_metrics=metrics_9892_1x1["persistence_metrics"],
            extra={
                "number_of_points": metrics_9892_1x1["number_of_centers"],
                "parameters": metrics_9892_1x1["number_of_parameters"],
                "best_epoch": metrics_9892_1x1["best_epoch"],
                "points_positive_MAE": (
                    summary_9892_1x1.get("points_with_positive_MAE_improvement")
                    if summary_9892_1x1 else None
                ),
                "points_negative_MAE": (
                    summary_9892_1x1.get("points_with_negative_MAE_improvement")
                    if summary_9892_1x1 else None
                ),
            },
        )

    # Tous points 5x5 masqué
    summary_5x5 = load_json(
        METRICS_DIR / "convlstm_all_points_masked_5x5_by_point_summary.json"
    )

    if summary_5x5:
        add_row(
            rows,
            experiment="ConvLSTM tous points 5x5 masqué",
            model="ConvLSTM",
            scale="9 892 points SAFRAN",
            patch="5x5 masqué",
            metrics=summary_5x5["global_convlstm_metrics"],
            persistence_metrics=summary_5x5["global_persistence_metrics"],
            extra={
                "number_of_points": summary_5x5["number_of_points"],
                "complete_5x5_points": summary_5x5["complete_5x5_points"],
                "incomplete_5x5_points": summary_5x5["incomplete_5x5_points"],
                "points_positive_MAE": summary_5x5["points_with_positive_MAE_improvement"],
                "points_negative_MAE": summary_5x5["points_with_negative_MAE_improvement"],
            },
        )

    dataframe = pd.DataFrame(rows)

    output_csv = METRICS_DIR / "final_results_table.csv"
    output_md = METRICS_DIR / "final_results_table.md"

    dataframe.to_csv(output_csv, index=False)

    def dataframe_to_markdown(df):
        columns = list(df.columns)

        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join(["---"] * len(columns)) + " |"

        rows_md = []
        for _, row in df.iterrows():
            values = []
            for column in columns:
                value = row[column]
                if pd.isna(value):
                    values.append("")
                elif isinstance(value, float):
                    values.append(f"{value:.4f}")
                else:
                    values.append(str(value))
            rows_md.append("| " + " | ".join(values) + " |")

        return "\n".join([header, separator, *rows_md])

    with output_md.open("w", encoding="utf-8") as file:
        file.write(dataframe_to_markdown(dataframe))

    print("=" * 80)
    print("TABLEAU FINAL DES RÉSULTATS")
    print("=" * 80)
    print(dataframe.to_string(index=False))

    print("\nFichiers créés :")
    print(f"- CSV      : {output_csv}")
    print(f"- Markdown : {output_md}")


if __name__ == "__main__":
    main()
