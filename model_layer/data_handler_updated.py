import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import config

class DataHandlerUpdated:
    """
    [UPDATED VERSION]
    Chuẩn bị dữ liệu thông minh:
    1. Tạo đặc trưng quá khứ (Lags).
    2. Tạo đặc trưng biến động (Rolling Stats).
    3. Chuẩn hóa dữ liệu.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_cols = None # Lưu tên cột để AI biết cái nào quan trọng

    def create_dataset(self, df, target_col='Spread_Z', lags=3, window=5):
        """
        Tạo bộ dữ liệu Input (X) và Output (y).
        Params:
            lags: Số ngày nhìn lại quá khứ.
            window: Cửa sổ tính thống kê trượt.
        """
        df = df.copy()
        
        # --- FEATURE ENGINEERING ---
        # Danh sách các cột cần tạo độ trễ
        cols_to_lag = [target_col, 'RSI', 'MACD', 'Dist_SMA']
        # Lọc chỉ lấy cột nào có trong df
        cols_to_lag = [c for c in cols_to_lag if c in df.columns]

        for col in cols_to_lag:
            # 1. Tạo cột Lag (Giá trị hôm qua, hôm kia...)
            for i in range(1, lags + 1):
                df[f'{col}_Lag_{i}'] = df[col].shift(i)
            
            # 2. Tạo cột Diff (Tốc độ thay đổi)
            df[f'{col}_Diff'] = df[col].diff()

        # 3. Tạo Rolling Stats cho Target (Bối cảnh thị trường)
        df[f'{target_col}_Roll_Mean'] = df[target_col].rolling(window=window).mean()
        df[f'{target_col}_Roll_Std'] = df[target_col].rolling(window=window).std()

        # --- TẠO TARGET ---
        # Dự báo Spread_Z của NGÀY MAI
        df['Target'] = df[target_col].shift(-1)
        
        df = df.dropna()
        
        X = df.drop(columns=['Target'])
        y = df['Target']
        
        self.feature_cols = X.columns
        return X, y

    def split_data(self, X, y):
        """
        Chia Train/Test và Scale dữ liệu.
        Trả về DataFrame (có tên cột) thay vì Numpy Array (mất tên).
        """
        split_point = int(len(X) * config.TRAIN_SPLIT)
        
        X_train = X.iloc[:split_point]
        X_test  = X.iloc[split_point:]
        y_train = y.iloc[:split_point]
        y_test  = y.iloc[split_point:]
        
        # Scaling
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled  = self.scaler.transform(X_test)
        
        # Chuyển ngược lại thành DataFrame để giữ tên cột
        X_train_df = pd.DataFrame(X_train_scaled, columns=self.feature_cols, index=X_train.index)
        X_test_df = pd.DataFrame(X_test_scaled, columns=self.feature_cols, index=X_test.index)
        
        return X_train_df, X_test_df, y_train, y_test