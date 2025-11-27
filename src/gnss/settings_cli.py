# src/gnss/settings_cli.py

from __future__ import annotations

from typing import Sequence

from gnss.settings import init_settings
from gnss.utils.probe_data import probe_data


def _prompt_str(prompt: str, default: str) -> str:
    s = input(f"{prompt} [{default}]: ").strip()
    return s or default


def _prompt_float(prompt: str, default: float) -> float:
    s = input(f"{prompt} [{default}]: ").strip()
    if not s:
        return default
    return float(s)


def _prompt_int(prompt: str, default: int) -> int:
    s = input(f"{prompt} [{default}]: ").strip()
    if not s:
        return default
    return int(s)


def _prompt_bool(prompt: str, default: bool) -> bool:
    d = "y" if default else "n"
    s = input(f"{prompt} (y/n) [{d}]: ").strip().lower()
    if not s:
        return default
    return s in ("y", "yes", "1")


def _prompt_prn_list(prompt: str, default: Sequence[int]) -> list[int]:
    default_str = ",".join(str(p) for p in default)
    s = input(f"{prompt} (comma separated) [{default_str}]: ").strip()
    if not s:
        return list(default)
    return [int(x) for x in s.replace("，", ",").split(",") if x.strip()]


def set_settings_cli():
    """
    命令行版 setSettings：
    1. 载入默认 settings（或你可以先自己调用 init_settings 自定义）
    2. 在终端中交互修改
    3. 可选运行 probe_data 预览原始数据
    4. 返回修改后的 settings
    """
    settings = init_settings()

    print("=== Signal / processing parameters ===")
    settings.fileName = _prompt_str("Data file", settings.fileName)
    settings.samplingFreq = _prompt_float("Sampling freq (Hz)", settings.samplingFreq)
    settings.skipNumberOfBytes = _prompt_int("N of bytes to skip", settings.skipNumberOfBytes)
    settings.dataType = _prompt_str("Data type", settings.dataType)
    settings.msToProcess = _prompt_int("N of ms to process", settings.msToProcess)
    settings.IF = _prompt_float("IF (Hz)", settings.IF)

    print("\n=== Acquisition settings ===")
    settings.acqSatelliteList = _prompt_prn_list(
        "Satellites (PRN) to be acquired", settings.acqSatelliteList
    )
    settings.acqSearchBand = _prompt_float("Acq. band (kHz)", settings.acqSearchBand)
    settings.acqThreshold = _prompt_float("Det. threshold", settings.acqThreshold)
    settings.skipAcquisition = int(
        _prompt_bool("Skip acquisition", bool(settings.skipAcquisition))
    )

    print("\n=== Tracking settings ===")
    settings.numberOfChannels = _prompt_int("Number of channels", settings.numberOfChannels)
    settings.plotTracking = int(
        _prompt_bool("Plot tracking", bool(settings.plotTracking))
    )

    print("\n--- PLL settings ---")
    settings.pllNoiseBandwidth = _prompt_float(
        "PLL bandwidth (Hz)", settings.pllNoiseBandwidth
    )
    settings.pllDampingRatio = _prompt_float(
        "PLL damping ratio", settings.pllDampingRatio
    )

    print("\n--- DLL settings ---")
    settings.dllNoiseBandwidth = _prompt_float(
        "DLL bandwidth (Hz)", settings.dllNoiseBandwidth
    )
    settings.dllDampingRatio = _prompt_float(
        "DLL damping ratio", settings.dllDampingRatio
    )
    settings.dllCorrelatorSpacing = _prompt_float(
        "DLL correlator spacing (chips)", settings.dllCorrelatorSpacing
    )

    print("\n=== Positioning settings ===")
    settings.navSolPeriod = _prompt_int("Nav. sol. period (ms)", settings.navSolPeriod)
    settings.elevationMask = _prompt_float("Elevation mask (deg)", settings.elevationMask)
    settings.useTropCorr = int(
        _prompt_bool("Troposphere correction", bool(settings.useTropCorr))
    )

    print("\nTrue receiver coordinates (UTM, can be NaN):")
    settings.truePosition.E = _prompt_float("E", settings.truePosition.E)
    settings.truePosition.N = _prompt_float("N", settings.truePosition.N)
    settings.truePosition.U = _prompt_float("U", settings.truePosition.U)

    print("\n配置已更新。是否先 Probe data 预览原始数据？")
    if _prompt_bool("Run probe_data now", True):
        probe_data(settings)

    print("\n设置已应用。")
    return settings


if __name__ == "__main__":
    # 允许直接 python -m gnss.settings_cli 运行
    set_settings_cli()
