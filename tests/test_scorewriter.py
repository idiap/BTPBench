# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

from pathlib import Path

import numpy
import pandas
import pytest

from btpbench.baselines import Template
from btpbench.scorewriter import CSVScoreWriter


def test_template_metadata_defaults_are_independent():
    first = Template("1", "first", None)
    second = Template("2", "second", None)

    first.metadata["session"] = "one"

    assert second.metadata == {}


def test_csv_writer():
    csv_file_path = Path("./test_score_file.csv")
    metadata_names = [
        "metadata_0",
        "metadata_1",
    ]

    csv_writer = CSVScoreWriter(csv_file_path, metadata_names)

    metadata_1 = dict()
    metadata_1["metadata_0"] = 10
    metadata_1["metadata_1"] = 11
    template_1 = Template("1", "0", None, metadata_1)

    metadata_2 = dict()
    metadata_2["metadata_0"] = 20
    metadata_2["metadata_1"] = 21
    template_2 = Template("2", "1", None, metadata_2)

    score = 42.0
    csv_writer.write_score(score, template_1, template_2)

    assert Path("./test_score_file.csv.temporary").exists()

    csv_writer.close()
    assert csv_file_path.exists()

    df = pandas.read_csv(csv_file_path)
    df_columns = list(df.columns.values)

    expected_colomn_names = [
        "probe_template_id",
        "probe_subject_id",
        "bio_ref_template_id",
        "bio_ref_subject_id",
        "score",
        "probe_metadata_0",
        "probe_metadata_1",
        "bio_ref_metadata_0",
        "bio_ref_metadata_1",
    ]
    for n in expected_colomn_names:
        assert n in df_columns

    assert df.shape[0] == 1

    row = df.iloc[0]
    assert row["probe_template_id"] == 1
    assert row["probe_subject_id"] == 2
    assert row["bio_ref_template_id"] == 0
    assert row["bio_ref_subject_id"] == 1
    assert row["probe_metadata_0"] == 20
    assert row["probe_metadata_1"] == 21
    assert row["bio_ref_metadata_0"] == 10
    assert row["bio_ref_metadata_1"] == 11
    assert row["score"] == 42

    csv_file_path.unlink()


def test_csv_writer_leaves_missing_metadata_empty(tmp_path):
    csv_file_path = tmp_path / "scores.csv"
    writer = CSVScoreWriter(csv_file_path, ["session", "camera"])
    reference = Template("1", "ref", None, {"session": "one"})
    probe = Template("2", "probe", None, {"camera": "front"})

    writer.write_score(0.5, reference, probe)
    writer.close()

    row = pandas.read_csv(csv_file_path).iloc[0]
    assert row["bio_ref_session"] == "one"
    assert numpy.isnan(row["bio_ref_camera"])
    assert numpy.isnan(row["probe_session"])
    assert row["probe_camera"] == "front"


def test_csv_writer_rejects_multiple_scores(tmp_path):
    writer = CSVScoreWriter(tmp_path / "scores.csv", [])
    reference = Template("1", "ref", None)
    probe = Template("2", "probe", None)

    with pytest.raises(TypeError, match="score must be a scalar"):
        writer.write_score([0.1, 0.2], reference, probe)

    writer.close()
