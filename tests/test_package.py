"""
Package-level checks: the public names, and that the implementation module
does not collide with the `bl2` instance (Phase 4 namespace fix).
"""
import importlib

import pytest

import bounded_lognorm
from bounded_lognorm import bl2, bounded_lognorm2_gen, PRECISION_DIAL


def test_public_names():
    assert isinstance(bl2, bounded_lognorm2_gen)
    assert bounded_lognorm.bl2 is bl2
    assert PRECISION_DIAL == ('fast', 'balanced', 'precise', 'exact')
    assert set(bounded_lognorm.__all__) >= {
        "bl2", "bounded_lognorm2_gen", "PRECISION_DIAL", "__version__"}


def test_no_bl2_submodule_shadowed_by_the_instance():
    """Before Phase 4, `bounded_lognorm.bl2` was both a submodule and the
    instance, and `import bounded_lognorm.bl2 as m` returned the instance.
    The module is now private (_bl2); the old path must fail loudly."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("bounded_lognorm.bl2")
    assert importlib.import_module("bounded_lognorm._bl2").bl2 is bl2


def test_version_is_a_release_string():
    parts = bounded_lognorm.__version__.split(".")
    assert len(parts) == 3 and all(p.isdigit() for p in parts)
