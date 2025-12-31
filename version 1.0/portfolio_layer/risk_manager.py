import numpy as np
import pandas as pd

class DynamicRiskManager:
    """
    [RISK CORE]
    Quản lý rủi ro động dựa trên sai số của mô hình AI (Model Residuals).
    Nguyên lý: Nếu gần đây AI dự báo sai nhiều -> Giảm độ tin cậy -> Giảm tỷ trọng.
    """
    def __init__(self, lookback_window=20):
        self.lookback = lookback_window

    def calculate_forecast_risk(self, df_history):
        """
        Tính toán rủi ro dự báo (Forecast Error Risk).
        
        Input: df_history cần có cột:
               - 'Spread': Giá trị thực tế
               - 'Spread_Pred': Giá trị AI đã dự báo (nếu có)
               
        Lưu ý: Nếu chưa có lịch sử dự báo ('Spread_Pred'), sẽ dùng độ lệch chuẩn của Spread gốc.
        """
        # Trường hợp 1: Có lịch sử dự báo -> Tính rủi ro dựa trên SAI SỐ
        if 'Spread_Pred' in df_history.columns:
            # Residual = Thực tế - Dự báo
            residuals = df_history['Spread'] - df_history['Spread_Pred']
            
            # Tính độ lệch chuẩn của sai số (Rolling Std Dev)
            rolling_std = residuals.rolling(window=self.lookback).std()
            current_sigma = rolling_std.iloc[-1]
            
            # Xử lý NaN
            if np.isnan(current_sigma):
                current_sigma = residuals.std()
                
        # Trường hợp 2: Chưa có lịch sử dự báo -> Dùng độ biến động của SPREAD gốc
        else:
            rolling_std = df_history['Spread'].rolling(window=self.lookback).std()
            current_sigma = rolling_std.iloc[-1]
            if np.isnan(current_sigma):
                current_sigma = df_history['Spread'].std()

        # Tính VaR (Value at Risk) 95% đơn giản
        # Giả định phân phối chuẩn: Risk = 1.65 * Sigma
        dynamic_risk = 1.65 * current_sigma
        
        # Đảm bảo không trả về NaN hoặc 0
        if np.isnan(dynamic_risk) or dynamic_risk == 0:
            return 1.0 # Giá trị mặc định an toàn
            
        return dynamic_risk

    def adjust_confidence(self, raw_prediction, dynamic_risk):
        """
        Điều chỉnh độ hấp dẫn (Expected Return) dựa trên rủi ro.
        Công thức Signal-to-Noise Ratio (SNR).
        """
        # Tránh chia cho 0
        risk_denominator = dynamic_risk + 1e-6
        
        # Tỷ lệ SNR: Tín hiệu / Nhiễu
        # SNR cao (Tín hiệu rõ, Nhiễu thấp) -> Giữ nguyên dự báo
        # SNR thấp (Tín hiệu yếu, Nhiễu cao) -> Giảm dự báo về 0
        snr = abs(raw_prediction) / risk_denominator
        
        # Hệ số phạt (Penalty): Kẹp trong khoảng [0, 1]
        # Nếu SNR > 1 (Tín hiệu > Nhiễu) -> Penalty = 1 (Không phạt)
        # Nếu SNR < 1 (Nhiễu > Tín hiệu) -> Penalty = SNR (Giảm tỷ trọng)
        penalty = np.clip(snr, 0.0, 1.0)
        
        # Lợi nhuận kỳ vọng đã điều chỉnh
        adjusted_return = raw_prediction * penalty
        
        return adjusted_return