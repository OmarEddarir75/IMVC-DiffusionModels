import importlib.util
from pathlib import Path

# Resolve repo root relative to this file: DCG (NEW)/check_loader_cli.py -> ../
repo = Path(__file__).resolve().parent.parent
file_path = repo / "DCG (NEW)" / "datasets.py"

spec = importlib.util.spec_from_file_location("dcg_new_datasets", str(file_path))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

main_dir = str((repo / "DCG (NEW)").resolve())
keys = ["cub", "landuse_21", "handwritten", "synthetic3d"]

print("main_dir=", main_dir)
for k in keys:
    cfg = {"dataset": k, "main_dir": main_dir}
    try:
        x_list, y_list = mod.load_data(cfg)
        print(f"OK {k}: x={[tuple(x.shape) for x in x_list]}, y={[tuple(y.shape) for y in y_list]}")
    except Exception as e:
        print(f"FAIL {k}: {type(e).__name__}: {e}")
