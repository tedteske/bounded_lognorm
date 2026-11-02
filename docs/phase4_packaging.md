# BL2 — Packaging for PyPI (Phase 4)

**Status:** released. `bounded-lognorm` 0.1.0 was published to PyPI on 1 Nov 2026 from tag `v0.1.0` (commit `493a912`); see §8. Full test suite: 434 passed, 18 skipped (Suite B, which needs the original implementation).

**Epistemic labels**, as in Phases 2 and 3: *Numerical* = measured on this machine or in CI; true for what was tested, not a guarantee beyond it.

---

## 1. Decisions

- **Distribution name `bounded-lognorm`, import name `bounded_lognorm`.** PyPI had no project under `bounded-lognorm`, `bounded_lognorm` (the same name after normalization) or `boundedlognorm`.
- **Version 0.1.0.** 0.x signals that the API may still change; Phase 3 changed it.
- **BL2 only.** The piecewise BL stays in `bounded_lognorm.ipynb`. It has closed-form CDF and quantile functions, which BL2 cannot have (Phase 1), so it is worth keeping. But it never went through Phases 2–3: its moments are the old biased grid sums, and it is only C¹ at the mode. The two are different distributions: the maximum CDF gap between them is 0.03–0.05 for typical cases and 0.26 with the mode near a bound (Numerical, 8 cases). It could ship later as `bl`, in its own `_bl.py`.
- **Branch.** Phase 4 was stacked on the PR #16 branch; #16 (Phases 1–3) and then #17 (Phase 4) were merged into master with merge commits. Releases are tagged from master.

## 2. Package layout

- `src/` layout: `src/bounded_lognorm/__init__.py` and `src/bounded_lognorm/_bl2.py`. Tests import the installed package, so a packaging mistake (a missing file) fails the tests instead of being hidden by the source tree.
- **Namespace fix** (Phase 3, §10). The implementation module is now private, `_bl2.py`, and `bounded_lognorm.bl2` is only the distribution instance, as `scipy.stats.norm` is. Previously `import bounded_lognorm.bl2 as m` returned the instance; it now raises `ModuleNotFoundError`. `tests/test_package.py` guards this.
- The public API is unchanged: `bl2`, `bounded_lognorm2_gen`, `PRECISION_DIAL`, plus `__version__`.

## 3. pyproject.toml

- Build backend: hatchling. The version is read from `__version__` in `__init__.py`, the single place it is set.
- License metadata in the PEP 639 form (`license = "MIT"`, `license-files`).
- `core-metadata-version = "2.4"`. Recent hatchling writes Core Metadata 2.5, and whether PyPI accepts 2.5 uploads was not confirmed at the time; 2.4 is all this package needs.
- Dependencies: `requires-python >=3.10`, `numpy>=1.23.5`, `scipy>=1.9`.
- The test dependencies are a PEP 735 dependency group (`pip install -e . --group test`, pip 25.1 or later), so they are not published as an extra.
- The sdist carries the source, tests, docs, README, CHANGELOG, ATTRIBUTION and LICENSE; the notebooks (about 5 MB) stay in the repository only. The wheel is the package plus LICENSE.

## 4. Minimum versions (Numerical)

- Python 3.10, NumPy 1.23.5, SciPy 1.9.3: full suite passes, from source and from the built wheel.
- SciPy 1.8.1: fails, at `_shape_info` (`_ShapeInfo` first appeared in SciPy 1.9). This confirms 1.9 as the floor.
- Python 3.13, NumPy 2.5.4, SciPy 1.18.1: full suite passes.
- Python 3.10 reached end of life in October 2026. It is kept because it costs nothing while CI covers it.
- Fragility to watch: `_ShapeInfo` is imported from a private SciPy module (`scipy.stats._distn_infrastructure`). The CI jobs on the latest SciPy are what will catch a move.

## 5. Tests

`tests/test_bl2.py` was converted from module-level checks to parametrized pytest functions.

- Every check is one pytest item, so a failure names its exact dial position and case. Gold values and tolerances were spliced verbatim from the old file; the numeric-literal tokens of the two files differ only in the deleted report-printing code.
- Counts: the old script's 422 real checks, plus 9 from splitting the two multi-input D6 checks per input, plus 3 package checks = 434. Suite B skips per case (18) unless a module `bl2_old` with the original implementation is importable.
- The gold CDF and the A3 errors are cached, so C1 no longer depends on Suite A having run first.
- Mutation-checked against the old script: `_SUPPORT_TOL` back to 1e-14, the `_updated_ctor_param` override removed, `_fitstart` removed, and a small drift in the fused exponent each fail the identical set of checks in both.
- `python tests/test_bl2.py` still works (it runs pytest on the file), but with the src layout the package must be installed first.

## 6. CI (`.github/workflows/tests.yml`)

- On every pull request and every push to master: Ubuntu with Python 3.10 to 3.15 on the latest dependencies, Ubuntu with Python 3.10 on the minimum NumPy and SciPy, and macOS and Windows on Python 3.13.
- The package is installed non-editable, so tests run against the installed copy.
- A build job builds the sdist and wheel, runs `twine check --strict`, installs the wheel in a clean environment and uploads `dist/` as an artifact.

## 7. Publishing (`.github/workflows/publish.yml`)

Trusted Publishing (OIDC): no API token is stored anywhere.

- **TestPyPI:** run the workflow by hand (Actions → publish → Run workflow).
- **PyPI:** publish a GitHub release whose tag is `v` plus the version (`v0.1.0`). The build job refuses a tag that does not match `__version__`.
- Both paths build once, run `twine check`, run the full suite against the built wheel, then upload that same artifact.

One-time setup (done 1 Nov 2026):

1. On TestPyPI, a trusted publisher for project `bounded-lognorm`: owner `tedteske`, repository `bounded_lognorm`, workflow `publish.yml`, environment `testpypi`. It was added as *pending* and became an ordinary publisher when the first upload created the project.
2. The same on PyPI, with environment `pypi`.
3. The `pypi` GitHub environment requires approval from `tedteske` (self-review allowed), so a release waits in the Actions tab until approved. The `testpypi` environment has no protection rule.

Releasing a new version:

1. Bump `__version__` in `src/bounded_lognorm/__init__.py`, and `version` and `date-released` in `CITATION.cff`; add a dated CHANGELOG entry; merge to master.
2. Optional dry run: Actions → publish → Run workflow (TestPyPI). The version must not already exist on TestPyPI.
3. Create a GitHub release on master with tag `v<version>`. Approve the `pypi` deployment when the run pauses.

A version can be uploaded to an index only once. If the TestPyPI dry run reveals a problem, re-test with a dev version (for example `0.1.0.dev1`) rather than reusing 0.1.0; TestPyPI and PyPI are separate, so 0.1.0 remains available for the real release.

## 8. Release record: 0.1.0 (Numerical)

- **CI on PR #17:** all 10 jobs passed (Ubuntu with Python 3.10–3.15 on the latest dependencies, Python 3.10 on the minimum dependencies, macOS and Windows on 3.13, and the build-and-check job).
- **TestPyPI dry run:** the manually triggered publish run uploaded the wheel and sdist with digital attestations. In Google Colab (Python 3.13, Colab's own NumPy and SciPy), installing from TestPyPI gave version 0.1.0 from the installed location and a mean of 5.872972571882167 for mode 5, σ 10 on (−5, 20), equal to the 30-digit reference. The full suite from the TestPyPI sdist, run against that install, gave 434 passed, 18 skipped.
- **PyPI release:** GitHub release `v0.1.0` at `493a912`; the publish run passed its tag check and the suite against the built wheel, waited for approval, and uploaded. https://pypi.org/project/bounded-lognorm/
- **Check after release:** `pip install bounded-lognorm` into a clean Python 3.12 environment gave 0.1.0 and the same mean to every printed digit. PyPI lists `Requires-Python >=3.10`, `numpy>=1.23.5`, `scipy>=1.9` and license `MIT`.

## 9. Still open

- From Phase 3: the mass not following the mode as mode → lower. (`fit_bounds` and the deep-tail `isf` are resolved for 0.2.0; see `docs/v0.2_tails_and_fit_bounds.md`.)
- `Applications_100m_Sprints.ipynb` imports `from bounded_lognorm import bounded_lognorm`, the piecewise BL, which this package does not provide. It ran against a module exported from the original notebook. *BL is ported in 0.2.0 as `bl` (`docs/phase5_bl_port.md`); the notebook's import now raises an ImportError with the migration, and updating the notebook is Phase 6.*
- ~~The Wang et al. DOI in ATTRIBUTION.md is still unverified.~~ Verified for 0.2.0: 10.3141/2315-07.
