"""Build the local API reference from runtime signatures and source docstrings."""

import inspect
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import NaviLib as nv


def main():
    modules = [nv.cleaning, nv.eda, nv.feature_engineering, nv.statistical_tests,
               nv.modeling, nv.evaluation, nv.quality, nv.timeseries, nv.theme,
               nv.reporting, nv._common]
    pages = ["# NaviLib API reference\n", f"Generated for version {nv.__version__}.\n",
             "Use `nv.help_map(query)` to search by purpose and `help(function)` in Python. "
             "Legacy aliases are listed in [MIGRATION.md](MIGRATION.md). "
             "Examples with `df`, `X`, or `y` assume the dataset described by that function.\n"]
    for mod in modules:
        pages.append(f"## {mod.__name__}\n")
        for name in getattr(mod, "__all__", []):
            obj = getattr(mod, name)
            if not callable(obj) or name != getattr(obj, "__name__", name):
                continue
            pages.extend([f"### {name}\n", f"```python\n{mod.__name__}.{name}{inspect.signature(obj)}\n```\n",
                          f"```text\n{inspect.getdoc(obj) or ''}\n```\n"])
    destination = ROOT / "docs" / "API.md"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text("\n".join(pages), encoding="utf-8")
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
