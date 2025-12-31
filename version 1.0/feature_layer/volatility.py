import pandas as pd
import config

class VolatilityIndicators:
    """
    Các chỉ báo đo độ biến động (Rủi ro).
    """
    def __init__(self):
        self.window = config.WINDOW_SIZE
        self.num_std = config.BB_STD_DEV

    def add_bollinger_bands(self, df):
        """
        Bollinger Bands: Dải băng biến động.
        """
        df = df.copy()
        sma = df['Adj Close'].rolling(window=self.window).mean()
        std = df['Adj Close'].rolling(window=self.window).std()
        
        # Dải trên và Dải dưới
        upper_band = sma + (std * self.num_std)
        lower_band = sma - (std * self.num_std)
        
        # %B: Vị trí của giá so với 2 dải băng (0 là dải dưới, 1 là dải trên)
        # Nếu > 1: Giá vọt ra ngoài dải trên. Nếu < 0: Giá thủng dải dưới.
        bandwidth = upper_band - lower_band
        df['Boll_Percent'] = (df['Adj Close'] - lower_band) / bandwidth
        
        # Độ rộng dải băng (Bandwidth): Báo hiệu sắp có biến động lớn nếu dải băng nhỏ lại
        df['Boll_Width'] = bandwidth / sma
        
        return df