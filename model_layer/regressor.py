
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

class LinearTrader:
    """
    Mô hình Hồi quy tuyến tính (Linear Regression).
    Đơn giản nhưng hiệu quả để tìm mối quan hệ tuyến tính.
    """
    def __init__(self):
        self.model = LinearRegression()

    def train(self, X_train, y_train):
        """
        Giai đoạn Học (Training):
        Máy tính tìm ra công thức y = ax + b phù hợp nhất.
        """
        self.model.fit(X_train, y_train)
        print(" Mô hình đã học xong!")

    def predict(self, X_data):
        """
        Giai đoạn Dự báo:
        Đưa dữ liệu đầu vào, máy tính trả về dự đoán Spread Z ngày mai.
        """
        return self.model.predict(X_data)

    def evaluate(self, y_true, y_pred):
        """
        Chấm điểm mô hình (Thi cử).
        """
        # MSE: Sai số bình phương trung bình (Càng nhỏ càng tốt)
        mse = mean_squared_error(y_true, y_pred)
        
        # RMSE: Sai số thực tế (Cùng đơn vị với dữ liệu gốc)
        rmse = np.sqrt(mse)
        
        # R2 Score: Độ chính xác (Càng gần 1 càng tốt, âm là dự đoán bừa)
        r2 = r2_score(y_true, y_pred)
        
        print(f"📊 Kết quả đánh giá Model:")
        print(f"   - RMSE (Sai số trung bình): {rmse:.4f}")
        print(f"   - R2 Score (Độ phù hợp): {r2:.4f}")
        
        return rmse, r2