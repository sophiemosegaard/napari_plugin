from pathlib import Path

import pandas as pd


def save_measurements(
    table: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)