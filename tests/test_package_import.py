"""Package import: moyamoya_cvr must be importable from a clean environment."""


def test_import_moyamoya_cvr():
    import moyamoya_cvr

    assert hasattr(moyamoya_cvr, "__version__")
