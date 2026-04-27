"""Sanity-check that the package and all sub-modules import cleanly."""


def test_package_imports():
    import protean  # noqa: F401


def test_agents_import():
    from protean.agents import MaxDamagePlayer  # noqa: F401


def test_encoding_import():
    from protean.encoding import encode_battle  # noqa: F401


def test_eval_import():
    from protean.eval import round_robin  # noqa: F401
