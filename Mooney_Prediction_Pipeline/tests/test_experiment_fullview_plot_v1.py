from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PLOT = (
    ROOT
    / "reports"
    / "v371_challenger"
    / "global_rds_active_v1"
    / "m1_order_delta_residual_v1"
    / "m1_order_delta_jit_actual_vs_predicted_fullview.png"
)


def test_current_experiment_fullview_plot_exists_and_is_nonempty():
    assert PLOT.exists()
    assert PLOT.stat().st_size > 10_000

