import pandas as pd
import config

class TrendIndicators:
    """
    Các chỉ báo xác định xu hướng chính của dòng tiền.
    """
    def __init__(self):
        self.fast = config.MACD_FAST      # 12
        self.slow = config.MACD_SLOW      # 26
        self.signal = config.MACD_SIGNAL  # 9

    def add_sma_distance(self, df):
        """
        Khoảng cách giá so với đường trung bình (SMA).
        """
        # Copy để không ảnh hưởng dữ liệu gốc
        df = df.copy()
        
        # SMA: Giá trung bình của 20 phiên gần nhất (hoặc theo config)
        sma = df['Adj Close'].rolling(window=config.WINDOW_SIZE).mean()
        
        # Tính % lệch: Giá hiện tại đang cao hơn hay thấp hơn bao nhiêu % so với trung bình?
        df['Dist_SMA'] = (df['Adj Close'] - sma) / sma
        
        return df

    def add_macd(self, df):
        """
        MACD: Moving Average Convergence Divergence
        (Đường trung bình động hội tụ phân kỳ)
        """
        df = df.copy()
        price = df['Adj Close']
        
        # Đường nhanh (EMA 12) và Đường chậm (EMA 26)
        exp1 = price.ewm(span=self.fast, adjust=False).mean()
        exp2 = price.ewm(span=self.slow, adjust=False).mean()
        
        # Đường MACD = Nhanh - Chậm
        macd = exp1 - exp2
        
        # Đường Tín hiệu (Signal Line) = EMA 9 của đường MACD
        signal = macd.ewm(span=self.signal, adjust=False).mean()
        
        df['MACD'] = macd
        df['MACD_Signal'] = signal
        # Histogram: Khoảng cách giữa MACD và Signal (Đo độ mạnh xu hướng)
        df['MACD_Hist'] = macd - signal
        
        return df