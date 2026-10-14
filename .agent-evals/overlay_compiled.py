"""repoevals setup: copy the compiled modules of the installed dpnp release into the workspace (current directory),
next to the Python sources of the task's base. Run with `python -I` so the installed package is found, not the
workspace's sources."""
import importlib.util
import pathlib
import shutil

src = pathlib.Path(importlib.util.find_spec("dpnp").submodule_search_locations[0])
for f in [*src.rglob("*.so"), src / "_version.py"]:
    dst = pathlib.Path(f.relative_to(src.parent))
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(f, dst)
