import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller
import config  # Lấy tham số cấu hình (OUTLIER_THRESH)

class DataProcessor:
    """
    Class xử lý trung tâm:
    1. Làm sạch (Cleaning): Sửa lỗi Index, điền dữ liệu thiếu.
    2. Biến đổi (Transformation): Tính Log Returns (đưa về chuỗi dừng).
    3. Lọc nhiễu (Denoising): Kẹp giá trị (Winsorization).
    4. Đồng bộ (Alignment): Cắt dữ liệu theo khung thời gian chung.
    """
    def __init__(self):
        pass

    def _flatten_columns(self, df):
        """
        [Fix lỗi Yfinance] Chuyển MultiIndex columns về dạng đơn.
        VD: ('Adj Close', 'VCB.VN') -> 'Adj Close'
        """
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df

    def _fill_missing_values(self, df):
        """
        [Từ CleaningBasic] Điền dữ liệu thiếu.
        - Dùng 'time' interpolation: Tốt nhất cho Time Series liên tục.
        - Dùng ffill/bfill: Để trám nốt các lỗ hổng ở đầu/cuối chuỗi.
        """
        # Interpolate theo thời gian (tuyến tính)
        df = df.interpolate(method='time', limit_direction='both')
        # Fill nốt những chỗ còn lại (thường là row đầu tiên)
        df = df.ffill().bfill()
        return df

    def _compute_log_returns(self, df):
        """
        [Từ CleaningTimeSeries] Tính Log Returns.
        Công thức: R_t = ln(P_t / P_{t-1})
        Lợi ích: Có tính cộng, phân phối gần chuẩn hơn Simple Return.
        """
        df = self._flatten_columns(df)
        
        # Ưu tiên lấy Adj Close, nếu không có thì lấy Close
        price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
        
        if price_col not in df.columns:
            return pd.DataFrame() # Trả về rỗng nếu lỗi

        df = df.copy()
        # Tính Log Return
        df['Log_Return'] = np.log(df[price_col] / df[price_col].shift(1))
        
        return df.dropna()

    def _winsorize_outliers(self, df, col_name='Log_Return', threshold=3.0):
        """
        [Từ CleaningBasic] Xử lý nhiễu bằng Z-score Clipping.
        Thay vì xóa dòng (mất dữ liệu ngày), ta kẹp giá trị về biên (Threshold).
        """
        if col_name not in df.columns:
            return df

        series = df[col_name]
        mean = series.mean()
        std = series.std()
        
        # Xác định biên trên/dưới
        upper_bound = mean + threshold * std
        lower_bound = mean - threshold * std
        
        # Kẹp giá trị (Clip)
        df[col_name] = series.clip(lower=lower_bound, upper=upper_bound)
        return df

    def _check_stationarity(self, df, col_name='Log_Return'):
        """
        [Từ CleaningTimeSeries] Kiểm tra tính dừng (ADF Test).
        Chỉ in ra cảnh báo để User biết, không chặn luồng chạy.
        """
        if col_name not in df.columns: return

        try:
            result = adfuller(df[col_name].dropna())
            p_value = result[1]
            if p_value > 0.05:
                print(f"⚠️ Cảnh báo: {col_name} có thể KHÔNG DỪNG (p-value={p_value:.4f})")
        except Exception:
            pass # Bỏ qua nếu lỗi tính toán (dữ liệu quá ngắn)

    def _align_data(self, data_dict):
        """
        [Logic cốt lõi] Đồng bộ thời gian giữa các mã.
        Chỉ giữ lại những ngày mà TẤT CẢ các mã đều có dữ liệu.
        """
        if not data_dict: return {}

        # Tìm giao (Intersection) của index
        common_index = None
        valid_tickers = []

        for ticker, df in data_dict.items():
            if df.empty: continue
            if common_index is None:
                common_index = df.index
            else:
                common_index = common_index.intersection(df.index)
            valid_tickers.append(ticker)
        
        if common_index is None or len(common_index) == 0:
            print("❌ Lỗi: Không tìm thấy ngày giao dịch chung giữa các mã!")
            return {}

        # Cắt dữ liệu theo index chung
        aligned_dict = {}
        for t in valid_tickers:
            aligned_dict[t] = data_dict[t].loc[common_index].copy()
            
        print(f"-> Đã đồng bộ dữ liệu: {len(common_index)} phiên giao dịch chung.")
        return aligned_dict

    def process_all(self, raw_data_dict):
        """
        Hàm Main gọi bởi main.py.
        Pipeline: Fix lỗi -> Fill NA -> Log Return -> Lọc Nhiễu -> Check Dừng -> Đồng bộ.
        """
        processed_temp = {}
        
        print("⚙️ Đang xử lý dữ liệu (Cleaning & Transforming)...")

        for ticker, df in raw_data_dict.items():
            # 1. Fix lỗi cột & Fill NA
            df = self._flatten_columns(df)
            df = self._fill_missing_values(df)
            
            # 2. Tính Log Return
            df = self._compute_log_returns(df)
            
            if not df.empty:
                # 3. Lọc nhiễu (Outliers)
                df = self._winsorize_outliers(df, threshold=config.OUTLIER_THRESH)
                
                # 4. Kiểm tra tính dừng (Optional check)
                self._check_stationarity(df)
                
                processed_temp[ticker] = df
        
        # 5. Đồng bộ thời gian (Bước quan trọng nhất cho Pair Trading)
        final_data = self._align_data(processed_temp)
        
        return final_data