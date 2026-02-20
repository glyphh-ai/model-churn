"""Test that the encoder config is valid and roles encode correctly."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from encoder import ENCODER_CONFIG


def test_config_has_required_fields(encoder_config):
    """EncoderConfig must have dimension, seed, and layers."""
    assert encoder_config.dimension > 0
    assert encoder_config.seed >= 0
    assert len(encoder_config.layers) >= 1


def test_semantic_layer_exists(encoder_config):
    """Must have a semantic layer with identity and context segments."""
    layer_names = [l.name for l in encoder_config.layers]
    assert "semantic" in layer_names

    semantic = next(l for l in encoder_config.layers if l.name == "semantic")
    seg_names = [s.name for s in semantic.segments]
    assert "identity" in seg_names
    assert "context" in seg_names


def test_metrics_layer_exists(encoder_config):
    """Must have a metrics layer with numeric roles."""
    layer_names = [l.name for l in encoder_config.layers]
    assert "metrics" in layer_names

    metrics = next(l for l in encoder_config.layers if l.name == "metrics")
    roles = metrics.segments[0].roles
    role_names = [r.name for r in roles]
    assert "logins" in role_names
    assert "support_cases" in role_names
    assert "defects" in role_names
    assert "feature_adoption" in role_names


def test_customer_id_is_key_part(encoder_config):
    """customer_id must be the key_part role for temporal identity."""
    semantic = next(l for l in encoder_config.layers if l.name == "semantic")
    identity = next(s for s in semantic.segments if s.name == "identity")
    cid_role = next(r for r in identity.roles if r.name == "customer_id")
    assert cid_role.key_part is True


def test_numeric_roles_have_config(encoder_config):
    """All metric roles must have NumericConfig with THERMOMETER encoding."""
    metrics = next(l for l in encoder_config.layers if l.name == "metrics")
    for role in metrics.segments[0].roles:
        assert role.numeric_config is not None, f"{role.name} missing numeric_config"
        assert role.numeric_config.bin_width > 0
        assert role.numeric_config.encoding_strategy.value == "thermometer"


def test_temporal_config(encoder_config):
    """Temporal source should be auto with auto signal type."""
    assert encoder_config.temporal_source == "auto"
    assert encoder_config.temporal_config is not None
    assert encoder_config.temporal_config.signal_type == "auto"


def test_test_customers_have_raw_metrics_only(test_customers):
    """Test data should only contain raw metrics — no risk_level or churn_driver."""
    for c in test_customers:
        assert "risk_level" not in c, f"{c['customer_id']} has risk_level — test data should be raw"
        assert "churn_driver" not in c, f"{c['customer_id']} has churn_driver — test data should be raw"
        assert "customer_id" in c
        assert "logins" in c
        assert "support_cases" in c
        assert "defects" in c
        assert "feature_adoption" in c
