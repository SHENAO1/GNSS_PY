import os

# 运行方式 python list_src_tree.py > src_tree.txt

# 要遍历的目录
ROOT = os.path.join("src", "gnss")

# 需要忽略的目录名
EXCLUDE_DIRS = {
    "__pycache__",
    ".git",
    ".venv",
    ".idea",
    ".vscode"
}

def list_directory_tree(start_path, indent=""):
    """
    仅遍历 src/gnss 下的代码结构，自动过滤掉 __pycache__ 等
    """

    try:
        entries = os.listdir(start_path)
    except PermissionError:
        print(f"{indent}[NOACCESS] {start_path}")
        return

    # 排序后遍历
    entries.sort()

    for entry in entries:
        full_path = os.path.join(start_path, entry)

        # 排除不需要的目录
        if entry in EXCLUDE_DIRS:
            continue

        if os.path.isdir(full_path):
            print(f"{indent}[DIR ] {entry}/")
            list_directory_tree(full_path, indent + "    ")
        else:
            print(f"{indent}[FILE] {entry}")


if __name__ == "__main__":
    print(f"目录结构：{ROOT}\n")
    list_directory_tree(ROOT)
