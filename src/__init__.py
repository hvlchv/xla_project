"""
Public API cho thư viện adaptive_median_filter.
"""

# Lấy các bộ lọc từ file filters.py
from .filters import (
    standard_median_filter,
    two_pass_median_filter,
    AdaptiveTwoPassMedianFilter,
    EnhancedAdaptiveMedianFilter,
)

# Lấy hàm tạo nhiễu từ file noise.py
from .noise import add_impulse_noise

# Lấy hàm tính toán độ đo từ file metrics.py
from .metrics import all_metrics

# Lấy các hàm vẽ biểu đồ từ file visualize.py
from .visualize import (
    show_comparison,
    plot_all_metrics,
)

# Lấy hàm chạy thực nghiệm từ file benchmark.py
from .benchmark import run_benchmark