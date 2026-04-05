import os

PROJECT_ROOT = "."

def print_tree(root, max_depth=3, prefix=""):
    if max_depth < 0:
        return
    try:
        items = sorted(os.listdir(root))
    except Exception:
        return

    for item in items:
        path = os.path.join(root, item)
        print(prefix + "|-- " + item)
        if os.path.isdir(path):
            print_tree(path, max_depth - 1, prefix + "    ")

def preview_file(filepath, max_lines=80):
    print(f"\n===== {filepath} =====")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    print("... (truncated)\n")
                    break
                print(line.rstrip())
    except Exception as e:
        print(f"Error reading {filepath}: {e}")

def main():
    print("\n===== PROJECT STRUCTURE =====\n")
    print_tree(PROJECT_ROOT)

    key_files = [
        "ICDM.py",
        "baseModels.py",
        "loss.py",
        "util.py",
        "run.py",
        "configure.py"
    ]

    for file in key_files:
        if os.path.exists(file):
            preview_file(file)

if __name__ == "__main__":
    main()