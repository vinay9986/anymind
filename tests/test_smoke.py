from anymind.runtime.orchestrator import Orchestrator


def test_orchestrator_instantiates() -> None:
    orchestrator = Orchestrator()
    if orchestrator is None:
        raise AssertionError("Orchestrator should instantiate")
