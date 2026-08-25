from __future__ import annotations

import numpy as np
import pandas as pd

from hrp_lab.data.wrds_export import load_crsp_export
from hrp_lab.evaluation.metrics import compare_strategies, expected_shortfall, max_drawdown


def test_crsp_export_normalizes_ciz_columns_and_duplicate_rows(tmp_path):
    dates = pd.date_range("2024-01-02", periods=4, freq="B")
    rows = []
    for permno, ticker in ((10001, "AAA"), (10002, "BBB")):
        for offset, date in enumerate(dates):
            rows.append(
                {
                    "PERMNO": permno,
                    "Ticker": ticker,
                    "DlyCalDt": date.date().isoformat(),
                    "DlyRet": 0.001 * (offset + 1),
                    "DlyCap": 1_000_000 + permno,
                }
            )
    rows.append(dict(rows[0]))
    path = tmp_path / "crsp.csv.gz"
    pd.DataFrame(rows).to_csv(path, index=False, compression="gzip")

    panel = load_crsp_export(path, minimum_coverage=1.0)

    assert panel.returns.shape == (4, 2)
    assert panel.duplicate_rows == 2
    assert panel.labels == {10001: "AAA", 10002: "BBB"}
    assert np.isfinite(panel.returns.to_numpy()).all()


def test_tail_metrics_use_positive_loss_magnitude():
    returns = np.array([0.03, 0.02, 0.01, -0.01, -0.10])
    assert expected_shortfall(returns, 0.80) == 0.10
    assert max_drawdown(returns) > 0.10


def test_comparison_reports_positive_improvement_for_lower_tail_loss():
    baseline = np.array([0.01] * 95 + [-0.10] * 5)
    candidate = np.array([0.01] * 95 + [-0.05] * 5)
    comparison = compare_strategies(candidate, baseline)
    assert comparison["tail_risk_reduction_pct"] == 50.0

