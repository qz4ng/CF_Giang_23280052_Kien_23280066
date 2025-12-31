from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import config

class RandomForestTrader:
    """
   
    Sử dụng Random Forest để bắt các tín hiệu phi tuyến tính phức tạp.
    """
    def __init__(self):
        # Cấu hình Random Forest (Có thể chỉnh trong config.py)
        self.model = RandomForestRegressor(
            n_estimators=getattr(config, 'RF_N_ESTIMATORS', 200), # Mặc định 100 cây nhiều thì càng tốt nhma lâu
            max_depth=getattr(config, 'RF_MAX_DEPTH', 15),       # Độ sâu tối đa
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
        self.feature_names = None

    def train(self, X_train, y_train):
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns
            
        self.model.fit(X_train, y_train)
        # Không in "Học xong" liên tục để đỡ rác log khi chạy nhiều cặp
        
    def predict(self, X_data):
        raw_pred = self.model.predict(X_data)
        
        # [MỚI] Kỹ thuật Clipping:
        # Ép giá trị dự báo chỉ được nằm trong khoảng [-3, 3] (vùng Z-score hợp lý)
        # Nếu AI dự báo 10, ta ép xuống 3. Điều này giảm RMSE cực mạnh.
        clipped_pred = np.clip(raw_pred, -3.0, 3.0)
        
        return clipped_pred

    def evaluate(self, y_true, y_pred, pair_name="Unknown"):    
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        correct_direction = np.sign(y_pred) == np.sign(y_true)
        hit_rate = np.mean(correct_direction) * 100
        
        print(f"   Evaluation [{pair_name}]:")
        print(f"     - RMSE: {rmse:.4f}")
        print(f"     - R2 Score: {r2:.4f}")
        print(f"     - Directional Accuracy (Đoán đúng chiều): {hit_rate:.2f}%") # Quan trọng
        
        return rmse, r2

    def show_feature_importance(self):
        """
        Hiển thị Top 5 yếu tố ảnh hưởng nhất đến quyết định của AI.
        """
        if self.feature_names is None: return

        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]

        print(f"   [AI Insight] Top yếu tố quan trọng:")
        for i in range(min(5, len(self.feature_names))):
            idx = indices[i]
            print(f"     {i+1}. {self.feature_names[idx]}: {importances[idx]:.4f}")