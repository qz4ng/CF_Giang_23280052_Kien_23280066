import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from statsmodels.regression.rolling import RollingOLS # Cần cho Beta động
from itertools import combinations
import config

class PairsIndicatorsUpdated:
    """
    [UPDATED VERSION]
    1. Hỗ trợ lấy Top N cặp tốt nhất.
    2. Sử dụng Rolling Beta để tính Spread (Thích nghi thị trường).
    """
    def __init__(self):
        pass

    def find_top_n_pairs(self, data_dict: dict, top_n=5):
        """
        Quét và trả về danh sách Top N cặp đồng tích hợp tốt nhất.
        """
        tickers = list(data_dict.keys())
        # Danh sách lưu kết quả: [(p_value, pair_tuple), ...]
        scored_pairs = []
        
        print(f" [UPDATED] Đang quét đồng tích hợp để tìm Top {top_n}...")

        # Tạo tất cả tổ hợp
        for t1, t2 in combinations(tickers, 2):
            s1 = data_dict[t1]['Adj Close']
            s2 = data_dict[t2]['Adj Close']
            
            # Làm sạch dữ liệu chung
            df_temp = pd.concat([s1, s2], axis=1).dropna()
            
            # Bỏ qua nếu dữ liệu quá ngắn
            if len(df_temp) < 100: 
                continue
            
            try:
                # Kiểm định Cointegration
                score, pvalue, _ = coint(df_temp.iloc[:, 0], df_temp.iloc[:, 1])
                
                # Chỉ lấy những cặp đạt chuẩn P-value (theo config)
                if pvalue < config.COINT_PVALUE_THRESH:
                    scored_pairs.append((pvalue, (t1, t2)))
            except:
                continue
        
        # Sắp xếp danh sách theo P-value tăng dần (Càng nhỏ càng tốt)
        scored_pairs.sort(key=lambda x: x[0])
        
        # Lấy Top N
        top_pairs = [item[1] for item in scored_pairs[:top_n]]
        top_pvalues = [item[0] for item in scored_pairs[:top_n]]
        
        return top_pairs, top_pvalues

    def calculate_rolling_spread(self, df1: pd.DataFrame, df2: pd.DataFrame, window=60):
        """
        Tính Spread với Beta trượt (Rolling Beta).
        Spread sẽ luôn dao động quanh 0 tốt hơn so với Beta tĩnh.
        """
        # Chuẩn bị biến
        y = df1['Adj Close']
        x = df2['Adj Close']
        x = sm.add_constant(x)
        
        # Chạy Rolling OLS
        rolling_model = RollingOLS(y, x, window=window, min_nobs=window)
        rolling_res = rolling_model.fit()
        
        # Lấy Beta theo thời gian (Hệ số góc)
        beta_series = rolling_res.params.iloc[:, 1]
        
        # Fill Beta những ngày đầu (bị NaN) bằng giá trị đầu tiên tính được
        beta_series = beta_series.bfill()
        
        # Tính Spread: Y - Beta_t * X
        spread = df1['Adj Close'] - beta_series * df2['Adj Close']
        
        # Tính Z-score động (theo mean/std trượt của Spread)
        spread_mean = spread.rolling(window=window).mean()
        spread_std = spread.rolling(window=window).std()
        
        z_score = (spread - spread_mean) / spread_std
        
        # Đóng gói dữ liệu
        df_combined = pd.concat([df1.add_suffix('_Y'), df2.add_suffix('_X')], axis=1)
        df_combined['Spread'] = spread
        df_combined['Spread_Z'] = z_score
        df_combined['Beta'] = beta_series # Lưu để theo dõi
        
        return df_combined.dropna(), beta_series.mean()