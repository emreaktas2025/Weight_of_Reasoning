"""Test that all modules can be imported successfully."""

def test_imports():
    """Test importing all core modules."""
    import wor.core.runner
    import wor.core.utils
    import wor.metrics.activation_energy
    import wor.metrics.attention_entropy
    import wor.metrics.activation_path_length
    import wor.metrics.circuit_utilization_density
    import wor.metrics.stability_intermediate_beliefs
    import wor.metrics.feature_load
    import wor.metrics.rev_composite
    import wor.eval.evaluate
    import wor.eval.evaluate_phase3
    import wor.plots.plot_reports
    import wor.plots.plot_phase3
    import wor.stats.partial_corr
