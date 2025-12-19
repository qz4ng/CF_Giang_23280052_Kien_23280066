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
            max_depth=getattr(config, 'RF_MAX_DEPTH', 5),       # Độ sâu tối đa
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
        return self.model.predict(X_data)

    def evaluate(self, y_true, y_pred, pair_name="Unknown"):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        print(f"   Evaluation [{pair_name}]: RMSE={rmse:.4f}, R2={r2:.4f}")
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