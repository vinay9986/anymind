from types import SimpleNamespace

from anymind.config.schemas import ModelConfig, PricingConfig, SearchConfig
from anymind.runtime.usage import PricingTable, UsageTotals, extract_usage_from_messages


def test_usage_totals_and_extract() -> None:
    totals = UsageTotals()
    totals.add(3, 4)
    assert totals.input_tokens == 3
    assert totals.output_tokens == 4

    msg = SimpleNamespace(usage_metadata={"prompt_tokens": 2, "completion_tokens": 5})
    extracted = extract_usage_from_messages([msg])
    assert extracted.input_tokens == 2
    assert extracted.output_tokens == 5


def test_pricing_table_costs() -> None:
    config = PricingConfig(
        prices_per_1k_tokens={"model": {"input": 1.0, "output": 2.0}},
        default={"input": 0.5, "output": 1.5},
    )
    table = PricingTable(config)
    totals = UsageTotals(input_tokens=1000, output_tokens=500)
    cost = table.cost("model", totals)
    assert cost["input"] == 1.0
    assert cost["output"] == 1.0
    assert cost["total"] == 2.0


def test_pricing_table_prefix_and_default() -> None:
    config = PricingConfig(
        prices_per_1k_tokens={"gpt-": {"input": 2.0, "output": 4.0}},
        default={"input": 1.0, "output": 1.0},
    )
    table = PricingTable(config)
    totals = UsageTotals(input_tokens=500, output_tokens=500)
    prefix_cost = table.cost("gpt-4.1", totals)
    assert prefix_cost["input"] == 1.0
    default_cost = table.cost("other", totals)
    assert default_cost["input"] == 0.5


def test_model_config_defaults() -> None:
    model = ModelConfig(model="gpt-4.1")
    assert model.thread_id == "default"
    assert model.tools_enabled is True
    assert isinstance(model.search, (SearchConfig, type(None)))
