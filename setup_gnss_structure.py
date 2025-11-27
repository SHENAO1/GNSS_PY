import os
from pathlib import Path

def ensure_file(path: Path, content=""):
    """如果文件不存在，则创建文件并写入内容"""
    if not path.exists():
        path.write_text(content, encoding="utf-8")
        print(f"[CREATED] {path}")
    else:
        print(f"[EXISTS]  {path}")

def ensure_dir(path: Path):
    """如果目录不存在则创建"""
    path.mkdir(parents=True, exist_ok=True)
    print(f"[DIR OK]  {path}")

def setup_gnss_structure():
    project_root = Path(__file__).resolve().parent
    src_root = project_root / "src" / "gnss"

    # -------------------------------
    # 1. 创建顶层目录
    # -------------------------------
    ensure_dir(src_root)
    ensure_file(src_root / "__init__.py", "# GNSS package\n")

    # -------------------------------
    # 2. acquisition 模块
    # -------------------------------
    acq = src_root / "acquisition"
    ensure_dir(acq)
    ensure_file(acq / "__init__.py")
    ensure_file(acq / "ca_code.py", "# placeholder for CA code generation\n")
    ensure_file(acq / "acquisition_core.py", "# placeholder for acquisition core\n")

    # -------------------------------
    # 3. tracking 模块
    # -------------------------------
    tracking = src_root / "tracking"
    ensure_dir(tracking)
    ensure_file(tracking / "__init__.py")
    ensure_file(tracking / "tracking_core.py", "# placeholder for tracking core\n")

    tracking_loop = tracking / "loop"
    ensure_dir(tracking_loop)
    ensure_file(tracking_loop / "__init__.py")
    ensure_file(tracking_loop / "calc_loop_coef.py",
                "# placeholder for calcLoopCoef conversion\n")

    # -------------------------------
    # 4. navigation 模块
    # -------------------------------
    nav = src_root / "navigation"
    ensure_dir(nav)
    ensure_file(nav / "__init__.py")

    eph = nav / "ephemeris"
    ensure_dir(eph)
    ensure_file(eph / "__init__.py")
    ensure_file(eph / "ephemeris.py", "# placeholder for ephemeris decoding\n")

    ensure_file(nav / "skyplot.py", "# placeholder for sky plot\n")

    # -------------------------------
    # 5. utils 模块
    # -------------------------------
    utils = src_root / "utils"
    ensure_dir(utils)
    ensure_file(utils / "__init__.py")

    ensure_file(utils / "signal_utils.py", "# placeholder: signal helper functions\n")
    ensure_file(utils / "bit_utils.py", "# placeholder: bit/twos complement tools\n")
    ensure_file(utils / "math_utils.py", "# placeholder: math helpers\n")
    ensure_file(utils / "plot_utils.py", "# placeholder: debugging plots\n")

    print("\n========== GNSS Python 目录结构初始化完成 ==========")

if __name__ == "__main__":
    setup_gnss_structure()
