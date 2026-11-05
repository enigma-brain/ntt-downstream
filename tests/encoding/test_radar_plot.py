"""Unit tests for tokenizer radar plotting helpers."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for tests
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from ntt.evaluation.core.types import MetricCategory  # noqa: E402

from ntt_downstream.encoding.analysis.radar import (  # noqa: E402
    _load_reference_split,
    _to_known_category,
    load_report,
    plot_radar,
)


class TestPlotRadar:
    def test_uses_score_caps_and_hides_outer_spine(self):
        fig = plot_radar(
            {
                "tok_a": {"temporal": -1.0, "population": 0.25, "encoding": 1.4},
                "tok_b": {"temporal": -0.25, "population": 0.5, "encoding": 0.75},
            },
            legend_loc=None,
        )

        ax = fig.axes[0]

        assert ax.get_ylim() == (0.0, 1.5)
        # The floor (-0.5) sits at r=0; matplotlib >= 3.11 drops the polar
        # origin tick, so only the rings above it carry labels.
        assert [tick.get_text() for tick in ax.get_yticklabels()] == ["0.0", "0.5", "1.0"]
        assert not ax.spines["polar"].get_visible()

        tok_a_line = next(line for line in ax.get_lines() if line.get_label() == "tok_a")
        assert tok_a_line.get_ydata().tolist() == [0.0, 0.75, 1.5, 0.0]

        plt.close(fig)

    def test_places_first_category_at_top(self):
        fig = plot_radar({"tok": {"temporal": 0.0, "population": 0.5, "encoding": 1.0}}, legend_loc=None)

        ax = fig.axes[0]

        assert ax.get_theta_offset() == np.pi / 2
        assert ax.get_theta_direction() == -1.0

        plt.close(fig)

    def test_draws_solid_black_ceiling_and_pads_category_labels(self):
        fig = plot_radar(
            {"tok": {"temporal": 0.0, "population": 0.5, "encoding": 1.0}},
            category_label_pad=22,
            legend_loc=None,
        )

        ax = fig.axes[0]

        ceiling_line = next(line for line in ax.get_lines() if line.get_label() == "Perfect ceiling")
        assert ceiling_line.get_color() == "black"
        assert ceiling_line.get_linestyle() == "-"
        assert ax.xaxis.majorTicks[0].get_pad() == 22

        plt.close(fig)

    def test_can_disable_negative_floor(self):
        fig = plot_radar(
            {"tok": {"temporal": -0.25, "population": 0.5, "encoding": 1.25}},
            floor_negative=False,
            legend_loc=None,
        )

        ax = fig.axes[0]

        assert ax.get_ylim() == (0.0, 1.0)
        # r=0 (here 0.0) is the polar origin, which matplotlib >= 3.11 leaves unlabeled.
        assert [tick.get_text() for tick in ax.get_yticklabels()] == ["0.5", "1.0"]
        tok_line = next(line for line in ax.get_lines() if line.get_label() == "tok")
        assert tok_line.get_ydata().tolist() == [0.0, 0.5, 1.0, 0.0]

        plt.close(fig)

    def test_can_rename_or_hide_floor_ring(self):
        renamed_fig = plot_radar(
            {"tok": {"temporal": 0.0, "population": 0.5, "encoding": 1.0}},
            floor_name="Reference floor",
            legend_loc=None,
        )
        renamed_ax = renamed_fig.axes[0]

        assert any(line.get_label() == "Reference floor" for line in renamed_ax.get_lines())
        plt.close(renamed_fig)

        hidden_fig = plot_radar(
            {"tok": {"temporal": 0.0, "population": 0.5, "encoding": 1.0}},
            display_floor=False,
            legend_loc=None,
        )
        hidden_ax = hidden_fig.axes[0]

        assert all("floor" not in line.get_label().lower() for line in hidden_ax.get_lines())
        plt.close(hidden_fig)


class TestCategoryFallback:
    def test_unknown_category_falls_back_not_raises(self):
        """A genuinely-unknown category string never aborts the load.

        ``_to_known_category`` falls back to a valid ``MetricCategory`` value
        for an unrecognized category instead of raising.
        """
        valid = {m.value for m in MetricCategory}

        result = _to_known_category("totally_bogus_category")

        assert result in valid

    def test_known_new_category_passes_through(self):
        for value in (m.value for m in MetricCategory):
            assert _to_known_category(value) == value


class TestLoadReport:
    def test_loads_report_with_new_vocabulary(self, tmp_path):
        """A report-dict written in the paper-aligned category vocabulary loads and
        categorizes correctly without any pre-deserialization rewriting."""
        report_dict = {
            "results": {
                "acg_correlation": {
                    "name": "acg_correlation",
                    "category": "single_unit_statistics",
                    "direction": "higher",
                    "level": "per_unit",
                    "raw_value": 0.8,
                    "normalized_value": None,
                    "metadata": {},
                    "value_range": [None, None],
                    "source_data_type": None,
                },
                "population_corr": {
                    "name": "population_corr",
                    "category": "population_structure",
                    "direction": "higher",
                    "level": "population",
                    "raw_value": 0.6,
                    "normalized_value": None,
                    "metadata": {},
                    "value_range": [None, None],
                    "source_data_type": None,
                },
            },
            "category_scores": None,
            "skipped_metrics": {},
            "config": {},
            "original_data_type": None,
            "reconstructed_data_type": None,
        }
        path = tmp_path / "eval_report.json"
        path.write_text(json.dumps(report_dict))

        report = load_report(path)

        assert set(report.results) == {"acg_correlation", "population_corr"}
        assert report.results["acg_correlation"].category == MetricCategory("single_unit_statistics")
        assert report.results["population_corr"].category == MetricCategory("population_structure")
        assert report.results["acg_correlation"].raw_value == 0.8

    def test_load_report_unknown_category_degrades(self, tmp_path):
        """A genuinely-unknown category loads without raising (degrades to valid).

        ``EvaluationReport.from_dict`` raises ``ValueError`` on an unknown
        category. ``load_report`` must catch that, sanitize the offending category, and still return a
        report — the metric's other fields (``raw_value``/``direction``) are what
        the radar actually consumes.
        """
        valid = {m.value for m in MetricCategory}
        report_dict = {
            "results": {
                "weird_metric": {
                    "name": "weird_metric",
                    "category": "totally_bogus_xyz",
                    "direction": "higher",
                    "level": "population",
                    "raw_value": 0.5,
                    "normalized_value": None,
                    "metadata": {},
                    "value_range": [None, None],
                    "source_data_type": None,
                },
            },
            "category_scores": {"totally_bogus_xyz": 0.5},
            "skipped_metrics": {},
            "config": {},
            "original_data_type": None,
            "reconstructed_data_type": None,
        }
        path = tmp_path / "eval_report.json"
        path.write_text(json.dumps(report_dict))

        report = load_report(path)  # must not raise

        assert report.results["weird_metric"].category.value in valid
        assert report.results["weird_metric"].raw_value == 0.5


class TestLoadReferenceSplit:
    @staticmethod
    def _report_dict(category: str) -> dict:
        return {
            "results": {
                "m": {
                    "name": "m",
                    "category": category,
                    "direction": "higher",
                    "level": "population",
                    "raw_value": 0.5,
                    "normalized_value": None,
                    "metadata": {},
                    "value_range": [None, None],
                    "source_data_type": None,
                },
            },
            "category_scores": None,
            "skipped_metrics": {},
            "config": {},
            "original_data_type": None,
            "reconstructed_data_type": None,
        }

    def test_load_reference_split_unknown_category_does_not_abort(self, tmp_path):
        """An unknown category in the reference (Poisson/ceiling) report must not abort.

        ``_load_reference_split`` is called unwrapped inside ``RadarBundle.from_cache``,
        so a raise here would take down the whole bundle. It must degrade instead.
        """
        valid = {m.value for m in MetricCategory}
        artifact = {
            "aggregated": self._report_dict("totally_bogus_xyz"),
            "perfect_aggregated": self._report_dict("single_unit_statistics"),
        }
        path = tmp_path / "poisson_baseline.json"
        path.write_text(json.dumps(artifact))

        baseline, ceiling = _load_reference_split(path)  # must not raise

        assert baseline.results["m"].category.value in valid
        assert ceiling.results["m"].category == MetricCategory("single_unit_statistics")
