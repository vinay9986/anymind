from anymind.tools.core_tools import internet_search_enabled, register_core_tools


class FakeMCP:
    def __init__(self) -> None:
        self.registered: list[str] = []

    def tool(self):
        def _decorator(func):
            self.registered.append(func.__name__)
            return func

        return _decorator


def current_time() -> None:
    return None


def internet_search() -> None:
    return None


def pdf_extract_text() -> None:
    return None


def test_internet_search_enabled_requires_kagi_and_scrapfly() -> None:
    assert internet_search_enabled({"KAGI_API_KEY": "k", "SCRAPFLY_API_KEY": "s"})
    assert internet_search_enabled(
        {"KAGI_API_KEY": "k", "SCRAPFLY_API_KEY_SECRET_ARN": "arn:aws:secret"}
    )
    assert not internet_search_enabled({"KAGI_API_KEY": "k"})
    assert not internet_search_enabled({"SCRAPFLY_API_KEY": "s"})
    assert not internet_search_enabled({})


def test_register_core_tools_skips_internet_search_without_credentials() -> None:
    mcp = FakeMCP()
    register_core_tools(
        mcp,
        env={},
        current_time_tool=current_time,
        internet_search_tool=internet_search,
        pdf_extract_text_tool=pdf_extract_text,
    )
    assert mcp.registered == ["current_time", "pdf_extract_text"]


def test_register_core_tools_includes_internet_search_with_credentials() -> None:
    mcp = FakeMCP()
    register_core_tools(
        mcp,
        env={"KAGI_API_KEY": "k", "SCRAPFLY_API_KEY": "s"},
        current_time_tool=current_time,
        internet_search_tool=internet_search,
        pdf_extract_text_tool=pdf_extract_text,
    )
    assert mcp.registered == [
        "current_time",
        "internet_search",
        "pdf_extract_text",
    ]
