# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

from pathlib import Path

import numpy
import pandas
import pytest

from btpbench.metrics import (
    decimated_subject_neg_pos_scores,
    fpir_r1_thresh,
    fpirs_tpirss_threshss,
    neg_pos_scores,
    remove_nans,
    streamed_decimated_subject_neg_pos_scores,
    subject_neg_pos_scores,
    tpir_r1_thresh,
)


@pytest.fixture()
def score_file():
    return Path(__file__).parent.resolve() / "assets" / "scores-test.csv"


def test_metrics(score_file):
    df = pandas.read_csv(score_file)
    fta, new_df = remove_nans(df)

    assert fta == 0.01
    assert not new_df.isnull().values.any()

    neg_score, pos_score = neg_pos_scores(new_df)

    assert len(pos_score) == 29
    assert len(neg_score) == 268

    cmc_scores = subject_neg_pos_scores(new_df)
    assert len(cmc_scores) == 30

    decimated_cmc_scores = decimated_subject_neg_pos_scores(
        new_df, n_resamples=5, decimation_percentage=0.5
    )

    assert len(decimated_cmc_scores) == 5

    for tmp_cmc_scores in decimated_cmc_scores:
        n_empty = 0
        for tmp_neg, tmp_pos in tmp_cmc_scores:
            if len(tmp_pos) == 0:
                n_empty += 1

        assert n_empty / 30 >= 0.5
        assert n_empty / 30 <= 0.54

    assert fpir_r1_thresh(decimated_cmc_scores, -2)[0] == 1.0
    assert fpir_r1_thresh(decimated_cmc_scores, 0)[0] == 0.0

    assert tpir_r1_thresh(decimated_cmc_scores, -2)[0] >= 0.9
    assert tpir_r1_thresh(decimated_cmc_scores, 0)[0] == 0.0

    fpirs, tpirss, thresholdss = fpirs_tpirss_threshss(decimated_cmc_scores)

    assert tpirss.shape[0] == fpirs.shape[0]
    assert tpirss.shape[1] == 5

    assert thresholdss.shape[0] == fpirs.shape[0]
    assert thresholdss.shape[1] == 5


def test_streamed_decimation_uses_rank1_sufficient_statistics(tmp_path):
    score_file = tmp_path / "reference-major-scores.csv"
    cache_dir = tmp_path / "rank1-cache"
    rows = []
    subjects = [str(i) for i in range(6)]
    # The real identification writer emits every probe for one reference,
    # then repeats the reference cycle for each system key.
    for key_index in range(2):
        for reference_subject in subjects:
            for probe_subject in subjects:
                is_genuine = probe_subject == reference_subject
                score = 0.9 - 0.1 * key_index if is_genuine else 0.1
                if is_genuine and probe_subject == "5":
                    score = float("nan")
                rows.append(
                    {
                        "probe_template_id": f"probe-{probe_subject}",
                        "probe_subject_id": probe_subject,
                        "bio_ref_template_id": f"ref-{reference_subject}",
                        "bio_ref_subject_id": reference_subject,
                        "score": score,
                    }
                )
    pandas.DataFrame(rows).to_csv(score_file, index=False)

    fta, total_rows, streamed_scores = streamed_decimated_subject_neg_pos_scores(
        score_file,
        n_resamples=3,
        decimation_percentage=0.5,
        chunk_size=7,
        cache_dir=cache_dir,
    )

    assert fta == pytest.approx(2 / 72)
    assert total_rows == 72
    assert streamed_scores.n_resamples == 3
    assert streamed_scores.negative_maxima.shape == (3, 6)
    assert streamed_scores.positive_maxima.shape == (6,)
    assert streamed_scores.in_gallery.shape == (3, 6)

    # Three subjects are decimated. The subject whose genuine comparisons are
    # NaN is also open set unless it was already decimated.
    assert all((~mask).sum() in (3, 4) for mask in streamed_scores.in_gallery)
    assert fpir_r1_thresh(streamed_scores, 0.0)[0] == 1.0
    assert fpir_r1_thresh(streamed_scores, 0.2)[0] == 0.0
    assert tpir_r1_thresh(streamed_scores, 0.5)[0] == 1.0
    assert tpir_r1_thresh(streamed_scores, 1.0)[0] == 0.0

    fpirs, tpirss, thresholdss = fpirs_tpirss_threshss(streamed_scores)
    assert tpirss.shape == (len(fpirs), 3)
    assert thresholdss.shape == (len(fpirs), 3)

    assert len(list(cache_dir.glob("*.npz"))) == 1
    cached_fta, cached_rows, cached_scores = streamed_decimated_subject_neg_pos_scores(
        score_file,
        n_resamples=3,
        decimation_percentage=0.5,
        chunk_size=3,
        cache_dir=cache_dir,
    )
    assert cached_fta == fta
    assert cached_rows == total_rows
    assert numpy.array_equal(
        cached_scores.negative_maxima, streamed_scores.negative_maxima
    )
    assert numpy.array_equal(
        cached_scores.positive_maxima, streamed_scores.positive_maxima
    )
    assert numpy.array_equal(cached_scores.in_gallery, streamed_scores.in_gallery)
