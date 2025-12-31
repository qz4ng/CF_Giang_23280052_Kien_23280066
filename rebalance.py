import pandas as pd
import numpy as np
import datetime
import os
import pickle  # Thư viện để lưu "bộ não" xuống ổ cứng
import config_updated as config

# Import các module đã nâng cấp
from data_layer.loader import DataLoader
from data_layer.processor import DataProcessor
from feature_layer.clustering import MarketCluster
from feature_layer.pairs_updated import PairsIndicatorsUpdated
from feature_layer.trend import TrendIndicators
from feature_layer.momentum import MomentumIndicators
from feature_layer.volatility import VolatilityIndicators
from model_layer.data_handler_updated import DataHandlerUpdated
from model_layer.regressor_updated import RandomForestTrader

# Tạo thư mục để lưu model nếu chưa có
if not os.path.exists('models'):
    os.makedirs('models')

MODEL_PATH = 'models/current_portfolio_state.pkl'

def run_rebalance():
    print("\n" + "="*60)
    print(f" Tái cấu trúc - {datetime.date.today()}")
    print("="*60)

    # --- BƯỚC 1: CẬP NHẬT DỮ LIỆU MỚI NHẤT ---
    # Ghi đè ngày kết thúc trong config thành ngày hôm nay
    today_str = datetime.date.today().strftime('%Y-%m-%d')
    print(f"\n[1] Tải dữ liệu từ {config.START_DATE} đến {today_str}...")
    
    loader = DataLoader(config.START_DATE, today_str)
    raw_data = loader.download_data(config.TICKERS)
    
    processor = DataProcessor()
    processed_data = processor.process_all(raw_data)
    
    if not processed_data:
        print("Không có dữ liệu để chạy.")
        return

    # --- BƯỚC 2: QUÉT LẠI CẤU TRÚC THỊ TRƯỜNG (CLUSTERING) ---
    # Thị trường thay đổi, các nhóm ngành có thể vận động khác đi
    print(f"\n[2] Phân cụm lại thị trường (Clustering)")
    cluster_algo = MarketCluster(n_clusters=getattr(config, 'N_CLUSTERS', 4))
    cluster_map = cluster_algo.cluster_stocks(processed_data)
    
    # Gom nhóm
    clusters = {i: [] for i in range(getattr(config, 'N_CLUSTERS', 4))}
    for ticker, group_id in cluster_map.items():
        clusters[group_id].append(ticker)

    # --- BƯỚC 3: CHỌN CẶP MỚI (PAIR SELECTION) ---
    # Cặp VCB-BID tháng trước tốt, tháng này có thể hết đồng tích hợp
    # Ta phải chọn lại Top Pairs.
    print(f"\n[3] Chọn lọc lại các cặp tốt nhất ")
    
    pairs_logic = PairsIndicatorsUpdated()
    portfolio_state = [] # Danh sách chứa mọi thứ cần thiết để trade tháng sau
    
    for group_id, tickers in clusters.items():
        if len(tickers) < 2: continue
        
        group_data = {t: processed_data[t] for t in tickers}
        
        # Lấy Top 1 cặp tốt nhất mỗi nhóm (Hoặc Top N tùy config)
        top_pairs, _ = pairs_logic.find_top_n_pairs(group_data, top_n=1)
        
        for pair in top_pairs:
            print(f"   -> Cluster {group_id}: chọn cặp {pair}")
            
            # Tính toán lại toàn bộ chỉ báo cho dữ liệu mới nhất
            df1 = processed_data[pair[0]]
            df2 = processed_data[pair[1]]
            
            # Thêm chỉ báo (Trend, Mom, Vol)
            trend = TrendIndicators(); mom = MomentumIndicators(); vol = VolatilityIndicators()
            for df in [df1, df2]:
                df = trend.add_macd(df)
                df = trend.add_sma_distance(df)
                df = mom.add_rsi(df)
                df = vol.add_bollinger_bands(df)
            
            # Tính Rolling Spread & Z-score
            # Quan trọng: Dữ liệu spread này đã bao gồm biến động mới nhất
            df_pair, avg_beta = pairs_logic.calculate_rolling_spread(
                df1, df2, window=getattr(config, 'ROLLING_WINDOW', 60)
            )
            
            # Lưu tạm thông tin
            portfolio_state.append({
                'tickers': pair,
                'data': df_pair,
                'group': group_id,
                'beta_avg': avg_beta
            })

    # --- BƯỚC 4: HUẤN LUYỆN LẠI AI (RETRAINING) ---
    # Model cũ đã lỗi thời, ta tạo model mới học dữ liệu đến tận ngày hôm nay
    print(f"\n[4] Retrain mô hình  với dữ liệu mới")
    
    for item in portfolio_state:
        pair_name = f"{item['tickers'][0]}-{item['tickers'][1]}"
        df = item['data']
        
        # Tạo Feature Engineering (Lags, Rolling...)
        handler = DataHandlerUpdated()
        X, y = handler.create_dataset(df, target_col='Spread_Z', lags=getattr(config, 'LAG_DAYS', 3))
        
        # Retrain trên TOÀN BỘ dữ liệu hiện có (Không chia test nữa vì ta đang chuẩn bị trade tương lai)
        # Hoặc vẫn chia test để validate, nhưng model cuối cùng nên học hết.
        # Ở đây ta học trên X_scaled
        X_scaled = handler.scaler.fit_transform(X) # Fit lại scaler theo dữ liệu mới
        
        # Khởi tạo model mới
        rf_model = RandomForestTrader()
        rf_model.train(X_scaled, y) # Học lại từ đầu
        
        # Lưu model và scaler vào item để đóng gói
        item['model'] = rf_model
        item['scaler'] = handler.scaler # Rất quan trọng: Phải lưu Scaler để scale dữ liệu ngày mai
        item['feature_cols'] = handler.feature_cols # Lưu tên cột để khớp dữ liệu
        
        # Xóa dữ liệu df nặng để tiết kiệm bộ nhớ khi lưu (chỉ cần Model & Scaler là đủ trade)
        # Nhưng nếu Allocator cần tính Covariance lịch sử, ta nên giữ lại 60 dòng cuối
        item['recent_history'] = df.tail(100) 
        del item['data'] # Xóa df gốc đi
        
        print(f"   Cập nhật cho cặp {pair_name}")

    # --- BƯỚC 5: LƯU TRẠNG THÁI (SERIALIZATION) ---
    print(f"\n[5] Lưu trạng thái Portfolio xuống '{MODEL_PATH}'...")
    
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(portfolio_state, f)
        
    print(f"HOÀN TẤT!  sẵn sàng giao dịch cho tháng tới.")
    print("="*60)

if __name__ == "__main__":
    run_rebalance()