from pathlib import Path
import numpy as np

def load_raw_signal(file_path: str, num_samples: int) -> np.ndarray:
    """
    读取前端采集的原始数据（占位示例）。
    后面我们会根据你现在 Matlab 的数据格式（int8 / int16 / 复数）来改。
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到数据文件: {path}")

    # 这里只是示例：假设是 int8 实数数据
    data = np.fromfile(path, dtype=np.int8, count=num_samples)
    # 后续根据真实格式做 IQ 还原/归一化
    return data.astype(np.float32)