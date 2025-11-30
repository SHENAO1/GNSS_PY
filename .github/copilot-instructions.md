# GNSS-PY AI Coding Agent Instructions

## Project Overview

**GNSS_PY** is a Python-based software GNSS receiver implementing a complete signal processing pipeline: acquisition → tracking → navigation. The codebase is primarily a Python port/reference for the MATLAB SoftGNSS project, used for GNSS algorithm research and optimization.

**Key Purpose**: Signal processing research, DLL/PLL loop analysis, pseudorange computation, and MATLAB-Python cross-validation.

## Architecture & Data Flow

### Pipeline Stages
The processing happens in `src/gnss/post_processing.py` via this sequence:

1. **Acquisition** (`acquisition/`) → Find satellite PRN + frequency offset from 1-11ms signal
   - Input: Raw signal file (int8/int16)
   - Key functions: `acquisition_core.acquisition()`, correlation-based search
   - Output: `acqResults` dict with PRN, Doppler, code phase per satellite

2. **Tracking** (`tracking/`) → Continuously estimate code/phase for each channel
   - Input: Acquisition results + continuous raw signal stream
   - Key functions: `tracking_from_array()` processes blocks of signal
   - Key submodules:
     - `dll.py`: Code delay lock loop
     - `pll.py`: Carrier phase lock loop  
     - `loop_filter.py`: Second-order loop filters
   - Output: `trackResults` per channel with discriminators, errors, signals

3. **Navigation** (`navigation/`) → Extract nav messages, compute pseudoranges + position
   - Input: Tracking results (code/phase measurements)
   - Key functions: `post_navigation()`, `calculate_pseudoranges()`, `least_square_pos()`
   - Key submodules:
     - `nav_msg.py`: Decode 50-bit nav frames  
     - `ephemeris/`: Parse broadcast ephemeris, compute satellite positions
     - `positioning.py`: Least-squares solver for ECEF position
   - Output: Position (X/Y/Z, lat/lon/height), DOP values, per-channel azimuth/elevation

### Data Structures

**Settings** (`settings.py`): Dataclass mirroring MATLAB's `settings` struct
  - Sampling parameters: `samplingFreq`, `IF`, `codeFreqBasis`
  - Loop parameters: `dllNoiseBandwidth`, `pllNoiseBandwidth`, damping ratios
  - Control flags: `verboseTracking`, `plotTracking`, `saveTrackingResults`
  - Paths: `fileName`, `resultsDir`

**Acquisition Results**: Dict-like with PRN as key; each entry holds:
  - `carrierFreq`: Detected Doppler shift [Hz]
  - `codePhase`: Code phase offset [samples]
  - `peakMetric`: Correlation peak value

**Tracking Results**: List of `SimpleNamespace` objects (one per channel):
  - `.PRN`, `.status`: Satellite ID and lock status
  - `.absoluteSampleNum`: Current sample index  
  - `.code/carrierFreq`: Estimated frequencies
  - `.codeError`, `.carrierError`: Discriminator outputs
  - `.I/Q`: In-phase and quadrature components

**Navigation Results**: `SimpleNamespace` with numpy arrays indexed by epoch:
  - `.X/.Y/.Z`: Position ECEF [m]
  - `.latitude/.longitude/.height`: Position geodetic
  - `.channel.PRN`, `.channel.az/.el`: Per-satellite PRN, azimuth, elevation

## Critical Developer Workflows

### Running the Full Pipeline
```bash
# From workspace root:
python src/gnss/main.py
```
**Workflow**:
1. `main.py` calls `probe_data()` → plots raw IF signal spectrum (validate file/parameters)
2. Prompts user to confirm settings are correct
3. User enters "1" → triggers `post_processing()`
4. Full pipeline executes (acquisition → tracking → navigation)

### Configuration Before Running
Edit `src/gnss/settings.py` → `init_settings()` function:
- **Must set** `fileName` (path to binary signal file)
- **Must set** `IF`, `samplingFreq` (match your hardware frontend)
- Optional: `acqSatelliteList` (default: all 32 GPS PRNs), loop bandwidth parameters

### Tracking Control (Key Settings)
In `main.py`, configure:
- `settings.verboseTracking`: Print per-epoch progress (set `False` for large datasets)
- `settings.trackingPrintInterval`: Interval in ms (e.g., 5000 = print every 5 sec)
- `settings.plotTracking`: Generate tracking debug plots post-run (slow on large files)
- Windows only: Press **Q** during tracking to stop early (if `enableManualStopTracking=True`)

### GPU Acceleration (Optional)
- Set `settings.use_gpu_tracking = True` in `main.py` (requires CuPy installed)
- Falls back to CPU NumPy if CuPy unavailable
- Implemented in `tracking_core.py` via conditional imports

### Saving Results
```python
settings.saveTrackingResults = True
settings.resultsDir = "path/to/save"
```
Saves `.npz` file with pickled `trackResults`, `acqResults`, `settings`, `channel` objects.

## Project-Specific Conventions & Patterns

### Naming & MATLAB Compatibility
- **CamelCase for settings**: `samplingFreq`, `dllNoiseBandwidth` (not snake_case)
  - Reason: Direct correspondence with MATLAB code for cross-validation
- Field names in results objects mirror MATLAB (e.g., `trackResults[ch].PRN`, `.absoluteSampleNum`)
- Utility functions in `utils/` use descriptive names: `cart2utm()`, `least_square_pos()`

### SimpleNamespace + Dict Compatibility
Many functions accept either `SimpleNamespace` or `dict` objects (see `_get_field()` helper patterns):
```python
def _get_field(obj, name):
    if hasattr(obj, name):
        return getattr(obj, name)
    return obj[name]  # Try dict access
```
**Pattern**: Prefer `.` notation for Settings, but utilities must handle both.

### Numpy Array Initialization (Critical Bug Prevention)
**DO NOT** share numpy arrays across channels/epochs. Always create fresh arrays in loops:
```python
# ✅ CORRECT: New array per channel
results = []
for _ in range(n_ch):
    res = SimpleNamespace(code_error=np.zeros(1000))
    results.append(res)

# ❌ WRONG: All channels share same array
arr = np.zeros(1000)
results = [SimpleNamespace(code_error=arr) for _ in range(n_ch)]
```
See `tracking_core.py::_init_track_results()` for reference.

### Data Type Handling
Input signal dtype (int8, int16, int32, uint8, uint16) is specified in `settings.dataType` as string.
Helper function converts:
```python
def _dtype_from_string(s: str):
    mapping = {"int8": np.int8, "int16": np.int16, ...}
    return mapping[s.lower()]
```
Used in file I/O to avoid manual casting errors.

### Windows Console Integration
Only `tracking_core.py` uses `msvcrt` for keystroke detection (Windows only):
- Wraps import in try/except (graceful fallback on non-Windows)
- Checks `enableManualStopTracking` flag before listening
- Allows user to press **Q** to stop tracking early (useful for debugging long runs)

## Key Integration Points

### Entry Point: `main.py`
- Initializes settings via `init_settings()` (reading from `settings.py`)
- Calls `probe_data()` for exploratory visualization
- Routes to `post_processing()` on user confirmation

### Core Pipeline: `post_processing.py`
- Reads binary signal file into numpy array (respects `skipNumberOfBytes`)
- Calls `acquisition()` → returns `acqResults` dict
- Calls `pre_run()` → maps acquisitions to channels, returns channel init params
- Calls `tracking_from_array()` → processes signal in blocks, returns `trackResults`
- Calls `post_navigation()` → computes positions, DOP, sky plot
- Optionally saves results via `_save_tracking_results()` (as `.npz`)

### File I/O Conventions
- **Raw signals**: Binary int8/int16 files (big-endian or little-endian, configurable via dtype)
- **Results**: `.npz` (NumPy zip format) containing serialized Python objects
- Paths are absolute or relative to workspace root

## Common Debugging Patterns

1. **Signal validation**: `probe_data()` visualizes spectrum, time-domain, histogram
   - If no peaks → check `fileName`, `IF`, `samplingFreq` mismatch
   
2. **Acquisition issues**: Low peak metrics → increase `acqSearchBand` or check SNR
   
3. **Tracking lock**: Monitor `trackResults[ch].CN0` (carrier-to-noise ratio)
   - If erratic → loop bandwidth too high or SNR insufficient

4. **Position divergence**: Check `nav.DOP` values and # of valid satellites (>4 for 3D fix)

5. **Performance profiling**: Set `verboseTracking=False`, `plotTracking=False` for production runs

## Codebase Organization

```
src/gnss/
├── main.py           # Entry point
├── settings.py       # Configuration dataclass
├── post_processing.py # Main pipeline orchestrator
├── acquisition/      # Satellite search (Doppler × code phase)
├── tracking/         # Channel tracking loops (DLL/PLL)
├── navigation/       # Position solving & visualization
└── utils/            # Signal processing, I/O, plotting helpers
```

## Testing & Validation

- `tests/test.py` currently only validates CuPy import (minimal test coverage)
- Validation via:
  - Comparing MATLAB `.mat` outputs with Python `post_processing()` results
  - Plotting functions (`plot_acquisition.py`, `plot_tracking.py`, `plot_navigation.py`)
  - Manual inspection of skyplots and position fixes

---

**For AI agents**: Refer to MATLAB SoftGNSS documentation for algorithmic context (loop theory, ephemeris math). When modifying tracking/navigation logic, preserve SimpleNamespace/dict compatibility. Always validate settings before running pipeline (file existence, parameter ranges).
