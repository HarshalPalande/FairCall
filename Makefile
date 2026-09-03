.PHONY: install doctor venv-check train test demo backtest tamper-demo app clean clean-venv

# `?=` so both forms work:
#     make install VENV=$HOME/.venvs/arm     (command line)
#     export VENV=$HOME/.venvs/arm           (environment)
# Needed when the repo path contains a ":" — `python -m venv` refuses those, so
# the venv has to live outside the repo. See the guard in the install target.
VENV ?= .venv
PY   := $(VENV)/bin/python

# --- interpreter selection -------------------------------------------------
# requirements.txt is fully pinned, and those pins bound this repo to CPython
# 3.10-3.13. That range is the wheel matrix, not a preference:
#   numpy 2.1.3        requires >=3.10; wheels stop at cp313
#   pandas 2.2.3       wheels stop at cp313
#   scikit-learn 1.5.2 wheels stop at cp313
#   matplotlib 3.9.4   wheels stop at cp313  <- on 3.14 pip falls back to a
#                      source build and dies compiling freetype
#   shap 0.46.0        wheels stop at cp312; on 3.13 pip builds it from source,
#                      which needs a C toolchain (see `make doctor`)
# So we pin the interpreter rather than loosening the pins - loosening would
# mean unpinning numpy, and every number in the README rests on these versions.
# 3.13 is tried first: it is what the README's reported numbers were produced
# on. 3.12 is the smoothest install - every pin has a prebuilt wheel there.
# Override with:  make install PYTHON=/full/path/to/python3.13
PYTHON := $(shell \
	for c in python3.13 python3.12 python3.11 python3.10 python3 python; do \
		p=$$(command -v $$c 2>/dev/null) || continue; \
		"$$p" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)' 2>/dev/null \
			&& { echo "$$p"; break; }; \
	done)

install:
	@case "$(VENV)" in /*|~*) venv_path="$(VENV)" ;; *) venv_path="$(CURDIR)/$(VENV)" ;; esac; \
	case "$$venv_path" in *:*) \
		printf '\n  ERROR  the venv path contains a colon:\n           %s\n\n' "$$venv_path"; \
		printf '  `python -m venv` refuses any path containing the PATH separator ":".\n'; \
		printf '  Your repo lives under a directory with a ":" in its name, so the\n'; \
		printf '  default in-repo .venv cannot be created. Two fixes:\n\n'; \
		printf '  1. Put the venv outside the repo (quickest - nothing else changes):\n'; \
		printf '         make install VENV=$$HOME/.venvs/ai-risk-manager\n'; \
		printf '         make demo    VENV=$$HOME/.venvs/ai-risk-manager\n'; \
		printf '     Add it to every make call, or export it once:\n'; \
		printf '         export VENV=$$HOME/.venvs/ai-risk-manager\n\n'; \
		printf '  2. Rename the parent directory to remove the colon (permanent fix):\n'; \
		printf '         mv "%s" "%s"\n\n' "$$(dirname "$(CURDIR)")" "$$(dirname "$(CURDIR)" | tr ':' '-')"; \
		exit 1;; esac
	@if [ -z "$(PYTHON)" ]; then \
		printf '\n  ERROR  no CPython 3.10-3.13 found on PATH.\n'; \
		printf '         your `python3` is %s\n\n' "$$(python3 -V 2>&1 || echo 'not installed')"; \
		printf '  requirements.txt is fully pinned, and numpy / pandas / scikit-learn /\n'; \
		printf '  matplotlib publish no wheels for 3.14+. pip would fall back to source\n'; \
		printf '  builds and die compiling freetype. Failing here instead.\n\n'; \
		printf '  Fix, any one of:\n'; \
		printf '    macOS   brew install python@3.13 && make install\n'; \
		printf '    Debian  sudo apt install python3.13 python3.13-venv && make install\n'; \
		printf '    other   make install PYTHON=/full/path/to/python3.13\n\n'; \
		exit 1; \
	fi
	@$(PYTHON) -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)' || { \
		printf '\n  ERROR  PYTHON=%s is %s.\n' "$(PYTHON)" "$$($(PYTHON) -V 2>&1)"; \
		printf '         This repo needs CPython 3.10-3.13 (see the pin notes in the Makefile).\n\n'; \
		exit 1; }
	@if [ "$$(uname -s)" = "Darwin" ] && [ ! -e "$$(brew --prefix 2>/dev/null)/opt/libomp/lib/libomp.dylib" ]; then \
		printf '\n  NOTE   libomp not found. XGBoost wheels on macOS link @rpath/libomp.dylib\n'; \
		printf '         and Apple clang does not ship it. Run `brew install libomp` now\n'; \
		printf '         (another shell is fine) or this install fails its smoke test.\n\n'; \
	fi
	@printf '==> venv interpreter: %s (%s)\n' "$(PYTHON)" "$$($(PYTHON) -V 2>&1)"
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -r requirements.txt
	@$(MAKE) --no-print-directory doctor

# --- post-install smoke test -----------------------------------------------
# Turns the known native-dependency failures into one actionable line instead
# of a pytest collection traceback. Real imports, not file probes, so it also
# catches conda/MacPorts installs and the shap C-extension case. The libomp
# branch is macOS-only; on Linux this degrades to a plain import smoke test.
define DOCTOR_PY
import importlib.util, platform, sys

MAC = platform.system() == "Darwin"

def die(*lines):
    sys.stdout.flush()
    sys.stderr.write("\n  " + "\n  ".join(lines) + "\n\n")
    raise SystemExit(1)

print("  python       %s  (%s)" % (platform.python_version(), sys.executable))

try:
    import xgboost
except Exception as e:
    msg = str(e)
    first = (msg.strip().splitlines() or [""])[0]
    if MAC and ("libomp" in msg or "libxgboost" in msg):
        die("FAIL  xgboost cannot load libxgboost.dylib: the OpenMP runtime is missing.",
            "",
            "      macOS-only. The xgboost wheel links @rpath/libomp.dylib and Apple",
            "      clang does not ship it. One command fixes it:",
            "",
            "          brew install libomp && make doctor",
            "",
            "      raw error: " + first)
    die("FAIL  import xgboost: " + first)
print("  xgboost      %s" % xgboost.__version__)

try:
    import shap
except Exception as e:
    die("FAIL  import shap: %s" % e,
        "",
        "      shap 0.46.0 publishes no cp313 wheel, so on Python 3.13 pip builds it",
        "      from source and silently drops its C extension when no compiler is",
        "      present. shap/explainers/_tree.py then fails on `from .. import _cext`,",
        "      and src/model.py needs TreeExplainer for the SHAP summary plot.",
        "",
        "      either   xcode-select --install              (macOS)",
        "               sudo apt install build-essential    (Debian)",
        "      or       make clean-venv && make install PYTHON=python3.12",
        "               (3.12 has a prebuilt wheel for every pin)")
print("  shap         %s" % shap.__version__)

try:
    # razorpay 1.4.2 imports pkg_resources, which warns on setuptools>=67.
    # requirements.txt pins setuptools<81 so the import still works; the warning
    # is expected noise, so keep it out of an otherwise clean smoke-test report.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import razorpay
except Exception as e:
    die("FAIL  import razorpay: %s" % e,
        "",
        "      razorpay 1.4.2 imports pkg_resources, removed in setuptools>=81.",
        "      requirements.txt pins setuptools<81 for exactly this reason; if you",
        "      are seeing this, something upgraded setuptools inside .venv.")
print("  razorpay     ok")

missing = [m for m in ("numpy", "pandas", "sklearn", "matplotlib", "streamlit", "pytest", "anthropic")
           if importlib.util.find_spec(m) is None]
if missing:
    die("FAIL  not installed: %s" % ", ".join(missing), "", "      re-run: make install")
print("  deps         ok")
print("  doctor       all good - next: make demo")
endef
export DOCTOR_PY

doctor:
	@$(PY) -c "$$DOCTOR_PY"

venv-check:
	@test -x $(PY) || { \
		printf '\n  no %s found - run: make install\n\n' "$(PY)"; exit 1; }

doctor train test demo backtest tamper-demo app: venv-check

train:
	$(PY) -m src.model

test:
	$(PY) -m pytest tests/ -v

demo: train
	$(PY) -m scripts.demo

backtest: train
	$(PY) -m scripts.backtest

# No `train` dependency: this exercises the audit chain only, so it runs in
# seconds on a cold clone and is safe to demo live.
tamper-demo:
	$(PY) -m scripts.tamper_demo

app:
	$(VENV)/bin/streamlit run app/streamlit_app.py

clean:
	rm -rf data/*.csv models/*.pkl models/*.json artifacts/*.png artifacts/*.json audit_log/*.jsonl

clean-venv:
	rm -rf $(VENV)
