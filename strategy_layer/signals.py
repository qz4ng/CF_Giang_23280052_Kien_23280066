import pandas as pd
import numpy as np
import config

class SignalLogic:
    """
    Logic sinh tín hiệu giao dịch dựa trên Z-score.
    Nguyên tắc: Mean Reversion (Đảo chiều về trung bình).
    """
    def __init__(self):
        self.entry_threshold = config.Z_ENTRY_THRESHOLD # Ngưỡng vào lệnh (1.0)
        self.exit_threshold = config.Z_EXIT_THRESHOLD   # Ngưỡng thoát lệnh (0.0)
        self.stop_loss = config.Z_STOP_LOSS             # Ngưỡng cắt lỗ (3.5)

    def generate_signals(self, df, col_name='Spread_Z'):
        """
        Input: DataFrame có cột 'Spread_Z' (Giá trị thực hoặc Dự báo).
        Output: DataFrame có thêm cột 'Signal' (1: Long, -1: Short, 0: Flat).
        """
        df = df.copy()
        # Đảm bảo cột mục tiêu là số (Sửa để dùng col_name thay vì cứng nhắc 'Spread_Z')
        # fix lỗi duplicate columns bằng cách chỉ lấy cột chúng ta cần
        if isinstance(df[col_name], pd.DataFrame):
            print(f"Cảnh báo: Phát hiện 2 cột tên '{col_name}'. Đang lấy cột cuối cùng.")
            series_z = df[col_name].iloc[:, -1] # Lấy cột mới nhất (thường là cái vừa rename)
        else:
            series_z = df[col_name]

        series_z = pd.to_numeric(series_z, errors='coerce')
        
        # Tạo cột Signal mặc định là 0 (Không làm gì)
        df['Signal'] = 0
        
        # Biến trạng thái: Đang giữ lệnh gì? (0: Không, 1: Long, -1: Short)
        current_position = 0
        
        signals = []
        
        # Duyệt qua từng ngày (Loop) để giữ trạng thái lệnh (Stateful)
        # (Cách này chậm hơn vectorization nhưng dễ hiểu logic Entry/Exit hơn)
        for z_score in df['Spread_Z']:
            
            # --- TRƯỜNG HỢP 1: ĐANG CẦM TIỀN (CHƯA CÓ LỆNH) ---
            if current_position == 0:
                # Nếu Z-score thấp quá (<-2) -> Sợi thun bị nén -> MUA (Long)
                if z_score < -self.entry_threshold:
                    current_position = 1
                # Nếu Z-score cao quá (>2) -> Sợi thun bị căng -> BÁN (Short)
                elif z_score > self.entry_threshold:
                    current_position = -1
            
            # --- TRƯỜNG HỢP 2: ĐANG GIỮ LỆNH MUA (LONG) ---
            elif current_position == 1:
                # Nếu Z-score đã hồi về mức 0 (hoặc cao hơn) -> Chốt lời
                if z_score >= -self.exit_threshold:
                    current_position = 0
                # Nếu Z-score giảm sâu quá mức chịu đựng -> Cắt lỗ
                elif z_score < -self.stop_loss:
                    current_position = 0
            
            # --- TRƯỜNG HỢP 3: ĐANG GIỮ LỆNH BÁN (SHORT) ---
            elif current_position == -1:
                # Nếu Z-score đã hồi về mức 0 (hoặc thấp hơn) -> Chốt lời
                if z_score <= self.exit_threshold:
                    current_position = 0
                # Nếu Z-score tăng cao quá mức chịu đựng -> Cắt lỗ
                elif z_score > self.stop_loss:
                    current_position = 0
            
            signals.append(current_position)
            
        df['Signal'] = signals
        
        # Shift(1): Tín hiệu hôm nay dùng để vào lệnh ngày mai
        # (Để tránh nhìn thấy tương lai)
        df['Position'] = df['Signal'].shift(1).fillna(0)
        
        return df