# main_updated.py
import pandas as pd
import numpy as np
import config_updated as config  # Import file config mới

# --- IMPORT CÁC MODULE ---
from data_layer.loader import DataLoader
from data_layer.processor import DataProcessor

# Feature Layer (Bao gồm các bản nâng cấp)
from feature_layer.clustering import MarketCluster
from feature_layer.pairs_updated import PairsIndicatorsUpdated # Dùng bản Updated
from feature_layer.trend import TrendIndicators
from feature_layer.momentum import MomentumIndicators
from feature_layer.volatility import VolatilityIndicators

# Model Layer (AI)
from model_layer.data_handler_updated import DataHandlerUpdated # Dùng bản Updated
from model_layer.regressor_updated import RandomForestTrader    # Dùng Random Forest

# Portfolio Layer (Quản lý vốn)
from portfolio_layer.allocator import StrategyAllocator

def run_advanced_system():
    print("\n" + "="*70)
    print(" Chạy mô hình")
    print("="*70)

    # --------------------------------------------------------------------------
    # BƯỚC 1: TẢI & XỬ LÝ DỮ LIỆU
    # --------------------------------------------------------------------------
    print("\n[1/6] tải và làm sạch dữ liệu")
    loader = DataLoader(config.START_DATE, config.END_DATE)
    raw_data = loader.download_data(config.TICKERS)
    
    processor = DataProcessor()
    processed_data = processor.process_all(raw_data)
    
    if len(processed_data) == 0:
        print(" Lỗi: Không có dữ liệu sau khi xử lý.")
        return

    # --------------------------------------------------------------------------
    # BƯỚC 2: PHÂN CỤM THỊ TRƯỜNG (CLUSTERING)
    # --------------------------------------------------------------------------
    # Mục tiêu: Tránh chọn toàn bộ cặp trong cùng 1 ngành (Rủi ro tập trung)
    print(f"\n[2/6] Phân cụm {len(processed_data)} mã thành {config.N_CLUSTERS} nhóm hành vi")
    
    cluster_algo = MarketCluster(n_clusters=config.N_CLUSTERS)
    cluster_map = cluster_algo.cluster_stocks(processed_data)
    
    # Tổ chức lại dictionary: {Group_ID: [List Tickers]}
    clusters = {i: [] for i in range(config.N_CLUSTERS)}
    for ticker, group_id in cluster_map.items():
        clusters[group_id].append(ticker)

    # --------------------------------------------------------------------------
    # BƯỚC 3: CHỌN CẶP TINH HOA & TÍNH FEATURE (PAIR SELECTION)
    # --------------------------------------------------------------------------
    print("\n[3/6] Chọn lọc cặp tốt nhất & Tính toán chỉ báo kỹ thuật")
    
    pairs_logic = PairsIndicatorsUpdated()
    trend = TrendIndicators()
    mom = MomentumIndicators()
    vol = VolatilityIndicators()
    
    # Danh sách chứa thông tin đầy đủ để đưa vào AI
    # Mỗi phần tử là 1 dict chứa: Data, Model, Handler, Tickers...
    portfolio_candidates = [] 
    
    for group_id, tickers_in_group in clusters.items():
        # Bỏ qua nhóm quá ít mã
        if len(tickers_in_group) < 2: continue
        
        # Lọc data chỉ của nhóm này
        group_data = {t: processed_data[t] for t in tickers_in_group}
        
        # Lấy Top 1 cặp tốt nhất trong nhóm này (để đại diện)
        # (Có thể sửa thành top_n=2 nếu muốn đa dạng hơn nữa)
        best_pairs, p_vals = pairs_logic.find_top_n_pairs(group_data, top_n=1)
        
        if not best_pairs:
            print(f"   Nhóm {group_id}: Không tìm thấy cặp đồng tích hợp nào.")
            continue
            
        pair = best_pairs[0]
        p_val = p_vals[0]
        print(f"   Nhóm {group_id}: Chọn cặp {pair} (p-value: {p_val:.5f})")
        
        # --- TÍNH TOÁN FEATURE KỸ THUẬT ---
        df1 = processed_data[pair[0]]
        df2 = processed_data[pair[1]]
        
        # Thêm RSI, MACD, Bollinger... cho từng mã lẻ TRƯỚC khi gộp
        for df in [df1, df2]:
            df = trend.add_macd(df)
            df = trend.add_sma_distance(df)
            df = mom.add_rsi(df)
            df = vol.add_bollinger_bands(df)
            
        # --- TÍNH SPREAD & Z-SCORE (DÙNG ROLLING BETA) ---
        # Đây là cải tiến quan trọng so với Static Beta
        df_pair, avg_beta = pairs_logic.calculate_rolling_spread(
            df1, df2, window=config.ROLLING_WINDOW
        )
        
        # Lưu vào danh sách chờ huấn luyện
        portfolio_candidates.append({
            'tickers': pair,
            'data': df_pair,
            'group': group_id,
            'beta': avg_beta
        })

    # --------------------------------------------------------------------------
    # BƯỚC 4: HUẤN LUYỆN AI (RANDOM FOREST)
    # --------------------------------------------------------------------------
    print(f"\n[4/6] Huấn luyện mô hình (Random Forest) cho {len(portfolio_candidates)} cặp")
    
    for item in portfolio_candidates:
        pair_name = f"{item['tickers'][0]}-{item['tickers'][1]}"
        df = item['data']
        
        # 1. Chuẩn bị dữ liệu (Tạo Lags, Rolling Stats...)
        handler = DataHandlerUpdated()
        X, y = handler.create_dataset(df, target_col='Spread_Z', lags=config.LAG_DAYS)
        
        # 2. Chia tập Train/Test & Scale
        X_train, X_test, y_train, y_test = handler.split_data(X, y)
        
        # 3. Khởi tạo & Train Model
        rf_model = RandomForestTrader()
        rf_model.train(X_train, y_train)
        
        # 4. Đánh giá sơ bộ
        print(f"   -> Đánh giá {pair_name}:")
        preds = rf_model.predict(X_test)
        rf_model.evaluate(y_test, preds, pair_name=pair_name)
        
        # 5. Lưu Model và Handler vào item để dùng cho bước Allocator
        item['model'] = rf_model
        item['handler'] = handler
        
        # (Optional) Xem AI đang quan tâm chỉ báo nào nhất
        # rf_model.show_feature_importance()

    # --------------------------------------------------------------------------
    # BƯỚC 5: PHÂN BỔ VỐN (PORTFOLIO OPTIMIZATION)
    # --------------------------------------------------------------------------
    print("\n[5/6] Tính toán tỷ trọng vốn tối ưu (Mean-Variance + Dynamic Risk)")
    
    # Khởi tạo Allocator có bật chế độ Quản lý rủi ro (risk_manager=True)
    allocator = StrategyAllocator(risk_manager=True)
    
    # Hàm này sẽ tự động:
    # 1. Dùng AI dự báo Z-score ngày tiếp theo
    # 2. Dùng Risk Manager đo độ tin cậy
    # 3. Dùng Optimizer tính tỷ trọng Sharpe tốt nhất
    final_weights = allocator.allocate_capital(portfolio_candidates)

    # --------------------------------------------------------------------------
    # BƯỚC 6: BÁO CÁO KẾT QUẢ
    # --------------------------------------------------------------------------
    print("\n" + "="*70)
    print(" KHUYẾN NGHỊ PHÂN BỔ DANH MỤC (PORTFOLIO ALLOCATION)")
    print("="*70)
    
    total_alloc = 0
    for i, item in enumerate(portfolio_candidates):
        pair_str = f"{item['tickers'][0]} & {item['tickers'][1]}"
        weight_pct = final_weights[i] * 100
        group_id = item['group']
        beta_val = item['beta']
        
        if weight_pct > 0.1: # Chỉ in những cặp có tỷ trọng đáng kể
            print(f" 🔹 Nhóm {group_id} | {pair_str:<20} | Tỷ trọng: {weight_pct:6.2f}% | Beta TB: {beta_val:.2f}")
            total_alloc += weight_pct
            
    print("-" * 70)
    print(f" Tổng tỷ trọng đầu tư: {total_alloc:.2f}% (Còn lại {100-total_alloc:.2f}% Tiền )")
    print("="*70)

if __name__ == "__main__":
    run_advanced_system()