from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import config

class DataHandler:
    """
    Chuẩn bị dữ liệu cho Machine Learning:
    1. Tạo Target (Mục tiêu dự báo).
    2. Chuẩn hóa (Scaling).
    3. Chia Train/Test.
    """
    def __init__(self):
        self.scaler = StandardScaler()

    def create_dataset(self, df, target_col='Spread_Z'):
        """
        Tạo bộ dữ liệu X (Đầu vào) và y (Mục tiêu).
        Mục tiêu: Dùng thông tin HÔM NAY để dự đoán Spread Z của NGÀY MAI.
        """
        df = df.copy()
        
        # Tạo cột Target 
        # Shift(-1) nghĩa là kéo giá ngày mai về dòng hôm nay.
        # Vì ta muốn máy học: "Nếu hôm nay chỉ số là A, thì ngày mai kết quả là B".
        df['Target'] = df[target_col].shift(-1)
        
        # Loại bỏ dòng cuối cùng bị NaN do shift (vì ngày mai chưa có giá)
        df = df.dropna()
        
        # Tách Features (X) và Target (y)
        # X là tất cả các cột trừ cột Target
        # (Bao gồm RSI, MACD, Spread_Z hiện tại...)
        X = df.drop(columns=['Target'])
        y = df['Target']
        
        return X, y

    def split_data(self, X, y):
        """
        Chia dữ liệu thành 2 phần: Học (Train) và Thi (Test).
        Không được tráo bài (shuffle=False) vì đây là chuỗi thời gian.
        """
        # Chia theo tỷ lệ trong config (VD: 80% đầu để học, 20% sau để kiểm tra)
        split_point = int(len(X) * config.TRAIN_SPLIT)
        
        X_train = X.iloc[:split_point]
        X_test  = X.iloc[split_point:]
        y_train = y.iloc[:split_point]
        y_test  = y.iloc[split_point:]
        
        # Chuẩn hóa dữ liệu (Scaling)
        # Giúp đưa RSI (0-100) và MACD (0.xxx) về cùng một hệ quy chiếu để Model dễ học.
        # Chỉ fit trên tập Train để tránh "nhìn trộm" dữ liệu Test (Data Leakage).
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled  = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test