import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from itertools import combinations
import config

class PairsIndicators:
    """
    Logic cốt lõi cho Pair Trading:
    1. Tìm cặp đồng tích hợp (Cointegration).
    2. Tính Spread và Z-score.
    """
    def __init__(self):
        pass

    def find_best_pair(self, data_dict):
        """
        Quét tất cả các cặp có thể để tìm cặp 'bài trùng' nhất.
        """
        tickers = list(data_dict.keys())
        best_pvalue = 1.0
        best_pair = None
        
        print(f"🔍 Đang quét đồng tích hợp cho {len(tickers)} mã cổ phiếu...")

        # Tạo tất cả tổ hợp cặp đôi (VD: VCB-BID, VCB-CTG...)
        for t1, t2 in combinations(tickers, 2):
            # Lấy dữ liệu giá đóng cửa đã làm sạch
            s1 = data_dict[t1]['Adj Close']
            s2 = data_dict[t2]['Adj Close']
            
            # Kiểm tra độ dài, nếu lệch nhau quá nhiều thì bỏ qua
            min_len = min(len(s1), len(s2))
            if min_len < 100: continue # Cần ít nhất 100 ngày để kiểm tra
            
            s1 = s1.iloc[-min_len:]
            s2 = s2.iloc[-min_len:]

            # KIỂM ĐỊNH COINTEGRATION (Engle-Granger Test)
            # Null Hypothesis: Không đồng tích hợp.
            # Nếu p-value < 0.05 => Bác bỏ Null => Có đồng tích hợp.
            try:
                score, pvalue, _ = coint(s1, s2)
                if pvalue < best_pvalue:
                    best_pvalue = pvalue
                    best_pair = (t1, t2)
            except:
                continue
        
        return best_pair, best_pvalue

    def calculate_spread_zscore(self, df1, df2):
        """
        Tính Spread (Khoảng cách) và Z-score (Độ lệch chuẩn hóa).
        Công thức: Spread = Y - Beta * X
        """
        # 1. Hồi quy tuyến tính tìm Beta (Hedge Ratio)
        # Giả sử: Giá df1 = Beta * Giá df2 + E
        x = df2['Adj Close']
        y = df1['Adj Close']
        x = sm.add_constant(x) # Thêm hằng số chặn (intercept)
        
        model = sm.OLS(y, x).fit()
        beta = model.params[1] # Hệ số góc
        
        # 2. Tính Spread (Phần dư - Residual)
        # Đây chính là khoảng cách thực tế giữa 2 cổ phiếu sau khi đã cân chỉnh Beta
        spread = df1['Adj Close'] - beta * df2['Adj Close']
        
        # 3. Tính Z-score của Spread
        # Z-score cho biết Spread đang lệch bao nhiêu Sigma so với trung bình
        window = config.WINDOW_SIZE
        spread_mean = spread.rolling(window=window).mean()
        spread_std = spread.rolling(window=window).std()
        
        z_score = (spread - spread_mean) / spread_std
        
        # 4. Gộp dữ liệu lại để trả về
        # Đổi tên cột để phân biệt
        df_target = df1.add_suffix('_Y') # Mã chúng ta muốn trade chính
        df_ref = df2.add_suffix('_X')    # Mã tham chiếu
        
        df_combined = pd.concat([df_target, df_ref], axis=1)
        df_combined['Spread'] = spread
        df_combined['Spread_Z'] = z_score
        
        return df_combined.dropna(), beta