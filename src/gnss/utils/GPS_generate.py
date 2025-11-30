import numpy as np
import matplotlib.pyplot as plt
import os

# ================= 配置区域 =================
# 想要做位置解算，通常需要较长数据。
# 采样率设为 4MHz 以节省磁盘空间 (对 L1 C/A 足够了)
FS = 4.092e6           
F_IF = 0.0             # 设为 0 表示生成基带 I/Q 信号 (零中频)，更通用
DURATION = 5.0         # 默认生成 5 秒 (仅供测试跟踪)，如需解星历请改为 30.0
OUTPUT_FILE = "gps_multi_sat_IQ.bin"

# 定义 4 颗卫星的参数 (模拟一个特定的几何分布)
# Doppler: 多普勒频移 (Hz)
# CodePhase: 码相位延迟 (Chips, 0-1023)
SATELLITES = [
    {'prn': 1,  'doppler': 1200,  'code_phase': 150, 'snr': -20},
    {'prn': 10, 'doppler': -800,  'code_phase': 850, 'snr': -21},
    {'prn': 24, 'doppler': 400,   'code_phase': 500, 'snr': -19},
    {'prn': 32, 'doppler': -2500, 'code_phase': 10,  'snr': -22}
]

# ================= 核心工具函数 =================
def generate_ca_code(prn_id):
    """生成 C/A 码序列"""
    g2_taps = {
        1: [2, 6], 10: [2, 9], 24: [4, 5], 32: [4, 9],
        # 如需更多卫星需补充此表
    }
    s1, s2 = g2_taps.get(prn_id, [2, 6])
    
    g1 = np.ones(10, dtype=int)
    g2 = np.ones(10, dtype=int)
    ca = []
    
    for _ in range(1023):
        ca.append(g1[-1] ^ g2[s1-1] ^ g2[s2-1])
        new_g1 = g1[2] ^ g1[9]
        new_g2 = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]
        g1 = np.roll(g1, 1); g1[0] = new_g1
        g2 = np.roll(g2, 1); g2[0] = new_g2
    return np.array(ca)

def get_nav_bit(time_sec):
    """
    模拟导航电文数据位 (50bps)
    这里简化为每 20ms 翻转一次的随机数据，
    注意：无法用于真实位置解算，仅供信号跟踪测试。
    """
    bit_idx = int(time_sec * 50)
    # 使用简单的哈希确保同一时间的比特是确定的
    np.random.seed(bit_idx) 
    return 1 if np.random.rand() > 0.5 else -1

# ================= 主生成逻辑 =================
def generate_and_preview():
    print(f"准备生成 {DURATION} 秒的数据...")
    print(f"采样率: {FS/1e6} MHz, 包含卫星 PRN: {[s['prn'] for s in SATELLITES]}")
    
    # 预计算所有卫星的 C/A 码 (BPSK格式 -1/1)
    ca_table = {}
    for sat in SATELLITES:
        ca_table[sat['prn']] = 2 * generate_ca_code(sat['prn']) - 1

    # 分块处理以节省内存 (每块 100ms)
    chunk_size = int(FS * 0.1) 
    total_samples = int(FS * DURATION)
    num_chunks = int(np.ceil(total_samples / chunk_size))
    
    # 用于可视化的缓存 (只存前 10ms)
    preview_data = []
    preview_len = int(FS * 0.01)

    # 打开文件准备写入 (复数 I/Q 对应 2个 float32)
    with open(OUTPUT_FILE, 'wb') as f:
        for i in range(num_chunks):
            # 当前块的时间轴
            t_start = i * (chunk_size / FS)
            t = np.arange(chunk_size) / FS + t_start
            
            # 初始化混合信号 (复数零)
            # 使用 Complex64 (I路实部, Q路虚部)
            mixed_signal = np.zeros(chunk_size, dtype=np.complex64)
            
            # 叠加每颗卫星
            for sat in SATELLITES:
                # 1. 计算当前时刻的相位
                # 码相位 = (时间 * 码率 + 初始偏移) % 1023
                code_idx = ((t * 1.023e6 + sat['code_phase']) % 1023).astype(int)
                current_code = ca_table[sat['prn']][code_idx]
                
                # 2. 生成导航电文 (简化版)
                # 为了速度，这里简化处理，假设整个 chunk 是同一个 bit (不严谨但够用)
                # 严谨做法需要对每个 sample 算 bit
                data_bit = get_nav_bit(t_start) 
                
                # 3. 生成载波 (复数指数信号 exp(j*w*t))
                # 包含中频 + 多普勒
                freq = F_IF + sat['doppler']
                carrier = np.exp(1j * 2 * np.pi * freq * t)
                
                # 4. 计算幅度
                amp = 10 ** (sat['snr'] / 20)
                
                # 5. 累加到总信号
                mixed_signal += amp * data_bit * current_code * carrier
            
            # 添加高斯白噪声 (复数噪声)
            # 假设噪声功率为 1 (0dB)
            noise = (np.random.randn(chunk_size) + 1j * np.random.randn(chunk_size)) / np.sqrt(2)
            final_signal = mixed_signal + noise
            
            # 存一点用于画图
            if len(preview_data) < preview_len:
                needed = preview_len - len(preview_data)
                preview_data.extend(final_signal[:needed])
            
            # 写入文件 (交错 I/Q: I, Q, I, Q ...)
            # numpy 的 tofile 对于 complex 会自动存为 real, imag, real, imag
            final_signal.astype(np.complex64).tofile(f)
            
            print(f"进度: {min((i+1)*chunk_size/FS, DURATION):.2f}s / {DURATION}s", end='\r')

    print("\n生成完成！缓存已就绪。")
    return np.array(preview_data)

# ================= 可视化与保存控制 =================
def visualize_and_decide(data):
    plt.figure(figsize=(12, 8))
    
    # 1. 时域图 (前 200 个点)
    plt.subplot(2, 2, 1)
    plt.plot(data[:200].real, label='I (Real)')
    plt.plot(data[:200].imag, label='Q (Imag)', alpha=0.7)
    plt.title("Time Domain (First 200 samples)")
    plt.legend()
    plt.grid(True)
    
    # 2. 功率谱密度 (PSD)
    plt.subplot(2, 2, 2)
    plt.psd(data, NFFT=1024, Fs=FS/1e6, Fc=F_IF/1e6, color='orange')
    plt.title("PSD (Spectrum)")
    
    # 3. 星座图 (IQ Plot)
    plt.subplot(2, 2, 3)
    plt.plot(data[:1000].real, data[:1000].imag, '.', markersize=1, alpha=0.3)
    plt.title("Constellation (Noise dominated)")
    plt.axis('equal')
    plt.grid(True)
    
    # 4. 直方图 (验证高斯分布)
    plt.subplot(2, 2, 4)
    plt.hist(data.real, bins=50, alpha=0.7, label='I')
    plt.hist(data.imag, bins=50, alpha=0.7, label='Q')
    plt.title("Histogram")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

    # 询问用户
    print("-" * 30)
    print(f"文件路径: {os.path.abspath(OUTPUT_FILE)}")
    # 计算文件大小
    file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"文件大小: {file_size_mb:.2f} MB")
    
    choice = input("是否保留此数据文件? (y/n): ").strip().lower()
    if choice != 'y':
        os.remove(OUTPUT_FILE)
        print("文件已删除。")
    else:
        print(f"文件已保留: {OUTPUT_FILE}")
        print("提示：这是一个 Complex64 (I/Q) 格式的二进制文件。")

# ================= 运行 =================
if __name__ == "__main__":
    preview = generate_and_preview()
    visualize_and_decide(preview)