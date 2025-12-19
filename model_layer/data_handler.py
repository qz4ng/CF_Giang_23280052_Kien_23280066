# FILE: model_layer/data_handler.py

from sklearn.preprocessing import StandardScaler
import pandas as pd
import config

class DataHandler:
    def __init__(self):
        self.scaler = StandardScaler()

    def create_dataset(self, df, target_col='Spread_Z', lags=3):
        """
        lags=3: Thêm thông tin của 3 ngày quá khứ.
        """
        df = df.copy()
        
        # 1. TẠO LAG FEATURES (Quan trọng)
        # Lấy giá trị Z-score của t-1, t-2, t-3 đưa vào dòng hiện tại t
        for i in range(1, lags + 1):
            col_name = f'{target_col}_Lag{i}'
            df[col_name] = df[target_col].shift(i)
        
        # 2. TẠO TARGET
        # Target là Z-score ngày mai (t+1)
        df['Target'] = df[target_col].shift(-1)
        
        # 3. LÀM SẠCH
        # Shift sẽ tạo ra NaN ở đầu (do Lag) và cuối (do Target), cần drop hết
        df = df.dropna()
        
        # Trong method create_dataset của data_handler.py
        # Trong method create_dataset của data_handler.py

        # Thay vì lấy tất cả, hãy lọc thủ công hoặc dùng Correlation
        # Ví dụ: Chỉ lấy Z-score quá khứ và RSI
        keep_columns = [
            'Spread_Z', 
            'Spread_Z_Lag1', 
            'Spread_Z_Lag2',
            'Spread_Z_Lag3', # Thêm thử Lag 3 xem sao
        
            
            
            'Target'# Mục tiêu
        ]

        # Chỉ giữ lại các cột có trong danh sách trên (nếu tồn tại trong df)
        valid_cols = [c for c in keep_columns if c in df.columns]
        df = df[valid_cols]

        # Sau đó mới drop Target để tạo X
        X = df.drop(columns=['Target'])
        # [Mẹo] Chỉ giữ lại các cột Numeric cho chắc ăn
        X = X.select_dtypes(include=['float64', 'int64'])
        
        y = df['Target']
        
        print(f"   -> Đã tạo bộ dữ liệu với {X.shape[1]} features (Bao gồm {lags} Lags).")
        return X, y

    def split_data(self, X, y):
        """Chia dữ liệu thành 2 phần: Học (Train) và Thi (Test).
        Không được tráo bài (shuffle=False) vì đây là chuỗi thời gian.
        """
       
        # Chia theo tỷ lệ trong config (VD: 80% đầu để học, 20% sau để kiểm tra)
        split_point = int(len(X) * config.TRAIN_SPLIT)
        
        X_train = X.iloc[:split_point]
        X_test  = X.iloc[split_point:]
        y_train = y.iloc[:split_point]
        y_test  = y.iloc[split_point:]
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled  = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test