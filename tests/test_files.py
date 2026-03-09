from __future__ import annotations
import pytest
from pandas.testing import assert_frame_equal

import typing

import pandas as pd

from ap_model_training.files import paths_dataframe_from_csv
from ap_model_training.utils import MONAI_KEYS

if typing.TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("header", [True, False], ids=["header", "no header"])
def test_paths_dataframe_from_csv(header: bool, tmp_path: Path) -> None:
    csv_path = tmp_path / "test_data.csv"

    expected_dataframe = pd.DataFrame(
        [[f"/path/to/image{i}.tif", f"/path/to/label{i}.tif"] for i in range(5)],
        columns=[MONAI_KEYS.IMAGE, MONAI_KEYS.LABEL],
    )

    expected_dataframe.to_csv(csv_path, header=header, index=False)
    dataframe = paths_dataframe_from_csv(csv_path)
    assert_frame_equal(dataframe, expected_dataframe)
