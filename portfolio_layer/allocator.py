import numpy as np
import pandas as pd
from .optimizer import PortfolioOptimizer
from .risk_manager import DynamicRiskManager
import config

class StrategyAllocator:
    """
    [BUSINESS LOGIC]
    Cầu nối trung tâm:
    Data -> AI Model -> Risk Manager -> Optimizer -> Allocation Weights
    """
    def __init__(self, risk_manager=True):
        self.optimizer = PortfolioOptimizer(risk_free_rate=getattr(config, 'RISK_FREE_RATE', 0.0))
        
        # Khởi tạo Risk Manager nếu được yêu cầu
        if risk_manager:
            self.risk_manager = DynamicRiskManager(lookback_window=20)
        else:
            self.risk_manager = None

    def allocate_capital(self, pairs_info_list):
        """
        Tính toán tỷ trọng vốn cho danh sách các cặp.
        
        Input: pairs_info_list (List of dicts):
            [
                { 'tickers': ('VCB', 'BID'), 'data': df, 'model': rf_model, 'handler': handler },
                ...
            ]
        Output:
            final_weights: Mảng tỷ trọng [0.2, 0.3, 0.5...]
        """
        expected_returns = []
        historical_spreads = []
        
        print(f"\n[ALLOCATOR] Đang tính toán phân bổ vốn cho {len(pairs_info_list)} cặp...")

        for item in pairs_info_list:
            df = item['data']
            model = item['model']
            handler = item['handler']
            
            # --- 1. CHUẨN BỊ DỮ LIỆU DỰ BÁO ---
            # Cần gọi create_dataset để tạo đủ Lags, Rolling features
            lags = getattr(config, 'LAG_DAYS', 3)
            X_full, _ = handler.create_dataset(df, target_col='Spread_Z', lags=lags)
            
            if X_full.empty:
                expected_returns.append(0)
                historical_spreads.append(df['Spread'].tail(60).values) # Fallback
                continue

            latest_X = X_full.iloc[[-1]]
            
            # Scale dữ liệu (dùng scaler đã fit từ trước)
            # Scale dữ liệu
            latest_X_scaled_arr = handler.scaler.transform(latest_X)
            
            # [FIX] Chuyển lại thành DataFrame để giữ tên cột, 
            latest_X_scaled = pd.DataFrame(
                latest_X_scaled_arr, 
                columns=handler.feature_cols, # Lấy lại tên cột đã lưu
                index=latest_X.index
            )
            
            # --- 2. AI DỰ BÁO (RAW PREDICTION) ---
            pred_z = model.predict(latest_X_scaled)[0]
            raw_attractiveness = abs(pred_z) # Độ lớn của tín hiệu (Magnitude
            # 3. AI Dự báo
            pred_z = model.predict(latest_X_scaled)[0]
            
            # --- 3. ĐIỀU CHỈNH THEO RỦI RO ĐỘNG (DYNAMIC RISK) ---
            if self.risk_manager:
                # Tính rủi ro hiện tại của cặp này
                risk_val = self.risk_manager.calculate_forecast_risk(df)
                
                # Điều chỉnh lợi nhuận kỳ vọng
                adjusted_attractiveness = self.risk_manager.adjust_confidence(raw_attractiveness, risk_val)
            else:
                adjusted_attractiveness = raw_attractiveness

            # Lọc nhiễu: Nếu độ hấp dẫn quá nhỏ (<0.5 Sigma), coi như không đáng vào lệnh
            if adjusted_attractiveness < 0.5:
                adjusted_attractiveness = 0.0
                
            expected_returns.append(adjusted_attractiveness)
            
            # --- 4. LẤY SPREAD ĐỂ TÍNH TƯƠNG QUAN (COVARIANCE) ---
            # Lấy 60 ngày gần nhất để phản ánh rủi ro hiện tại
            hist_spread = df['Spread'].tail(60).values
            historical_spreads.append(hist_spread)

        # --- 5. TỐI ƯU HÓA (OPTIMIZATION) ---
        expected_returns = np.array(expected_returns)
        
        # Xử lý độ dài historical_spreads không đều nhau (do nghỉ lễ, lỗi data...)
        if len(historical_spreads) > 0:
            min_len = min([len(s) for s in historical_spreads])
            # Cắt đuôi cho bằng nhau
            trimmed_spreads = [s[-min_len:] for s in historical_spreads]
            
            # Tạo Covariance Matrix
            spread_df = pd.DataFrame(trimmed_spreads).T
            cov_matrix = spread_df.cov().values
        else:
            # Fallback nếu không có dữ liệu spread
            cov_matrix = np.eye(len(expected_returns))

        # Kiểm tra: Nếu toàn bộ thị trường đều xấu (Returns = 0)
        if np.sum(expected_returns) == 0:
            print("   -> Thị trường không có cơ hội (Z-score thấp hoặc Rủi ro cao).")
            print("   -> Thị Trường xấu đừng có đầu tư nha hihi hoặc giữ nguyên danh mục cũ).")
            return np.zeros(len(pairs_info_list))
            
        # Gọi Optimizer giải bài toán Markowitz
        weights = self.optimizer.optimize_sharpe_ratio(expected_returns, cov_matrix)
        
        return weights