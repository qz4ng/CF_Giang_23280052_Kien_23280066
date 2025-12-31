import pandas as pd
import numpy as np
import config

class MomentumIndicators:
    """
    Các chỉ báo đo sức mạnh của đà tăng/giảm.
    """
    def __init__(self):
        self.rsi_window = config.RSI_WINDOW # số ngày tính RSI

    def add_rsi(self, df):
        """
        RSI: Relative Strength Index (Chỉ số sức mạnh tương đối).
        Thang đo từ 0 đến 100.
        """
        df = df.copy()
        delta = df['Adj Close'].diff() # Giá hôm nay - Giá hôm qua
        
        # Tách khoản lãi (gain) và lỗ (loss)
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_window).mean()
        
        # Tránh lỗi chia cho 0
        loss = loss.replace(0, np.nan) 
        
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Fill NaN đầu tiên bằng 50 (mức trung tính)
        df['RSI'] = df['RSI'].fillna(50)
        return df

    def add_roc(self, df, periods=5):
        """
        ROC: Rate of Change (Tốc độ thay đổi giá).
        """
        df = df.copy()
        prev_price = df['Adj Close'].shift(periods)
        # Giá thay đổi bao nhiêu % so với 5 ngày trước
        df['ROC'] = ((df['Adj Close'] - prev_price) / prev_price) * 100
        return df.fillna(0)