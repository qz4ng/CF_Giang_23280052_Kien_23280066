import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import config

class Visualizer:
    """
    Class chuyên vẽ biểu đồ hiệu suất và tín hiệu giao dịch.
    """
    def __init__(self):
        # Sử dụng style 'ggplot' cho đẹp
        plt.style.use('ggplot')

    def plot_performance(self, df):
        """
        Vẽ 2 biểu đồ:
        1. Đường cong vốn (Equity Curve)
        2. Spread Z-score và các điểm vào lệnh Mua/Bán
        """
        # Tạo khung hình (Figure) có 2 hàng, 1 cột
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # --- BIỂU ĐỒ 1: EQUITY CURVE (LÃI LỖ TÍCH LŨY) ---
        ax1.plot(df.index, df['Cumulative_PnL'], label='Tổng Lãi/Lỗ', color='blue', linewidth=2)
        ax1.fill_between(df.index, df['Cumulative_PnL'], alpha=0.1, color='blue')
        ax1.set_title('Hiệu Suất Chiến Lược (Cumulative PnL)', fontsize=14)
        ax1.set_ylabel('Lợi nhuận (Điểm Spread)')
        ax1.legend(loc='upper left')
        ax1.grid(True)

        # --- BIỂU ĐỒ 2: Z-SCORE & TÍN HIỆU ---
        # Vẽ đường Z-score dự báo
        ax2.plot(df.index, df['Spread_Z'], label='Spread Z-score (Dự báo)', color='gray', alpha=0.7)
        
        # Vẽ các đường kẻ ngang (Ngưỡng Entry/Exit)
        ax2.axhline(y=config.Z_ENTRY_THRESHOLD, color='red', linestyle='--', label='Ngưỡng Short (+2)')
        ax2.axhline(y=-config.Z_ENTRY_THRESHOLD, color='green', linestyle='--', label='Ngưỡng Long (-2)')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        # --- Đánh dấu điểm Mua/Bán ---
        # Tìm những điểm mà vị thế thay đổi
        # Position = 1 (Long), -1 (Short), 0 (Flat)
        
        # Lọc ra các điểm MUA (Long Entry)
        # Điều kiện: Hôm nay Position=1 và hôm qua Position=0
        long_entries = df[(df['Position'] == 1) & (df['Position'].shift(1) == 0)]
        ax2.scatter(long_entries.index, long_entries['Spread_Z'], 
                    marker='^', color='green', s=100, label='Vào Lệnh MUA (Long)', zorder=5)

        # Lọc ra các điểm BÁN (Short Entry)
        # Điều kiện: Hôm nay Position=-1 và hôm qua Position=0
        short_entries = df[(df['Position'] == -1) & (df['Position'].shift(1) == 0)]
        ax2.scatter(short_entries.index, short_entries['Spread_Z'], 
                    marker='v', color='red', s=100, label='Vào Lệnh BÁN (Short)', zorder=5)
        
        # Lọc ra các điểm THOÁT LỆNH (Exit)
        # Điều kiện: Hôm nay Position=0 và hôm qua khác 0
        exits = df[(df['Position'] == 0) & (df['Position'].shift(1) != 0)]
        ax2.scatter(exits.index, exits['Spread_Z'], 
                    marker='o', color='black', s=50, label='Chốt Lời/Cắt Lỗ', zorder=5)

        ax2.set_title('Tín Hiệu Giao Dịch trên Spread Z-score', fontsize=14)
        ax2.set_ylabel('Z-score')
        ax2.legend(loc='upper left')
        ax2.grid(True)

        plt.tight_layout()
        print(" Đang hiển thị biểu đồ...")
        plt.show()
    def plot_model_fit(self, y_test, y_pred):
        """
        Vẽ biểu đồ đánh giá độ chính xác của Mô hình AI.
        So sánh giữa Giá trị Thực tế (Actual) và Giá trị Dự báo (Predicted).
        """
        # Chuyển đổi sang numpy array nếu input là Series
        if isinstance(y_test, pd.Series):
            y_test = y_test.values
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # --- BIỂU ĐỒ 1: SCATTER PLOT (Điểm phân tán) ---
        # Trục ngang: Giá trị thực, Trục dọc: Giá trị dự báo
        # Nếu mô hình chuẩn 100%, các điểm sẽ nằm trên đường chéo màu đỏ
        ax1.scatter(y_test, y_pred, color='blue', alpha=0.5, label='Dữ liệu kiểm thử')
        
        # Vẽ đường chéo tham chiếu (Perfect Prediction Line)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Dự báo hoàn hảo')
        
        ax1.set_title('Tương quan: Thực tế vs Dự báo', fontsize=12)
        ax1.set_xlabel('Giá trị Thực (Actual Spread Z)')
        ax1.set_ylabel('Giá trị Dự báo (Predicted Spread Z)')
        ax1.legend()
        ax1.grid(True)

        # --- BIỂU ĐỒ 2: LINE CHART (So sánh trên Time Series) ---
        # Chỉ lấy 100 điểm cuối cùng để vẽ cho đỡ rối mắt
        display_len = 100
        if len(y_test) > display_len:
            y_test_sub = y_test[-display_len:]
            y_pred_sub = y_pred[-display_len:]
        else:
            y_test_sub = y_test
            y_pred_sub = y_pred

        x_axis = np.arange(len(y_test_sub))
        ax2.plot(x_axis, y_test_sub, label='Thực tế', color='black', linewidth=1.5)
        ax2.plot(x_axis, y_pred_sub, label='AI Dự báo', color='orange', linestyle='--', linewidth=1.5)
        
        ax2.set_title(f'So sánh chi tiết ({len(y_test_sub)} phiên gần nhất)', fontsize=12)
        ax2.set_xlabel('Phiên giao dịch')
        ax2.set_ylabel('Spread Z-score')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        print(" Đang hiển thị biểu đồ đánh giá Model...")
        plt.show()  