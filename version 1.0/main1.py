# main_updated.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import config_updated as config

# --- IMPORT CÁC MODULE CŨ & MỚI ---
from data_layer.loader import DataLoader
from data_layer.processor import DataProcessor

# Feature Layer
from feature_layer.clustering import MarketCluster
from feature_layer.pairs_updated import PairsIndicatorsUpdated
from feature_layer.trend import TrendIndicators
from feature_layer.momentum import MomentumIndicators
from feature_layer.volatility import VolatilityIndicators

# Model Layer
from model_layer.data_handler_updated import DataHandlerUpdated
from model_layer.regressor_updated import RandomForestTrader

# Portfolio Layer
from portfolio_layer.allocator import StrategyAllocator

# [MỚI] IMPORT CÁC LAYER CŨ ĐỂ BACKTEST
from strategy_layer.signals import SignalLogic
from strategy_layer.backtester import Backtester
from strategy_layer.visualizer import Visualizer

def run_advanced_system():
    print("\n" + "="*70)
    print(" 🚀 HỆ THỐNG QUANT TRADING FULL-STACK (AI + BACKTEST)")
    print("="*70)

    # --------------------------------------------------------------------------
    # BƯỚC 1: TẢI & XỬ LÝ DỮ LIỆU
    # --------------------------------------------------------------------------
    print("\n[1/7] Tải và làm sạch dữ liệu...")
    loader = DataLoader(config.START_DATE, config.END_DATE)
    raw_data = loader.download_data(config.TICKERS)
    
    processor = DataProcessor()
    processed_data = processor.process_all(raw_data)
    
    if len(processed_data) == 0:
        print("❌ Lỗi: Không có dữ liệu.")
        return

    # --------------------------------------------------------------------------
    # BƯỚC 2: PHÂN CỤM (CLUSTERING)
    # --------------------------------------------------------------------------
    print(f"\n[2/7] Phân cụm thị trường ({config.N_CLUSTERS} nhóm)...")
    cluster_algo = MarketCluster(n_clusters=config.N_CLUSTERS)
    cluster_map = cluster_algo.cluster_stocks(processed_data)
    
    clusters = {i: [] for i in range(config.N_CLUSTERS)}
    for ticker, group_id in cluster_map.items():
        clusters[group_id].append(ticker)

    # --------------------------------------------------------------------------
    # BƯỚC 3: CHỌN CẶP & TÍNH FEATURE
    # --------------------------------------------------------------------------
    print("\n[3/7] Chọn cặp tốt nhất & Tính chỉ báo kỹ thuật...")
    pairs_logic = PairsIndicatorsUpdated()
    trend = TrendIndicators(); mom = MomentumIndicators(); vol = VolatilityIndicators()
    
    portfolio_candidates = [] 
    
    for group_id, tickers in clusters.items():
        if len(tickers) < 2: continue
        group_data = {t: processed_data[t] for t in tickers}
        
        # Chọn 1 cặp tốt nhất mỗi nhóm
        best_pairs, p_vals = pairs_logic.find_top_n_pairs(group_data, top_n=1)
        
        if best_pairs:
            pair = best_pairs[0]
            print(f"   ✅ Nhóm {group_id}: {pair} (p-value: {p_vals[0]:.5f})")
            
            # Tính Feature
            df1, df2 = processed_data[pair[0]], processed_data[pair[1]]
            for df in [df1, df2]:
                df = trend.add_macd(df); df = trend.add_sma_distance(df)
                df = mom.add_rsi(df); df = vol.add_bollinger_bands(df)
            
            # Tính Rolling Spread
            df_pair, avg_beta = pairs_logic.calculate_rolling_spread(df1, df2, window=config.ROLLING_WINDOW)
            
            portfolio_candidates.append({
                'tickers': pair,
                'data': df_pair,
                'group': group_id,
                'beta': avg_beta
            })

    # --------------------------------------------------------------------------
    # BƯỚC 4: HUẤN LUYỆN AI (TRAINING)
    # --------------------------------------------------------------------------
    print(f"\n[4/7] Huấn luyện AI (Random Forest)...")
    
    for item in portfolio_candidates:
        df = item['data']
        handler = DataHandlerUpdated()
        X, y = handler.create_dataset(df, target_col='Spread_Z', lags=config.LAG_DAYS)
        
        # Chia Train/Test
        X_train, X_test, y_train, y_test = handler.split_data(X, y)
        
        # Train Model
        rf_model = RandomForestTrader()
        rf_model.train(X_train, y_train)
        
        # Lưu lại để dùng sau
        item['model'] = rf_model
        item['handler'] = handler
        item['X_test'] = X_test # Lưu lại để Backtest
        item['y_test'] = y_test

    # --------------------------------------------------------------------------
    # BƯỚC 5: TÍNH TỶ TRỌNG VỐN (ALLOCATION)
    # --------------------------------------------------------------------------
    print("\n[5/7] Tính toán tỷ trọng vốn (Portfolio Optimization)...")
    allocator = StrategyAllocator(risk_manager=True)
    
    # Lấy tỷ trọng tối ưu cho TƯƠNG LAI (dựa trên dữ liệu cuối cùng)
    final_weights = allocator.allocate_capital(portfolio_candidates)

    # --------------------------------------------------------------------------
    # BƯỚC 6: BACKTEST HIỆU SUẤT TỪNG CẶP (TÍCH HỢP MODULE CŨ)
    # --------------------------------------------------------------------------
    print("\n[6/7] 🔄 CHẠY BACKTEST TRÊN DỮ LIỆU KIỂM THỬ (TEST SET)...")
    
    # Khởi tạo các module cũ
    sig_gen = SignalLogic()
    backtester = Backtester()
    visualizer = Visualizer()
    
    portfolio_daily_returns = pd.DataFrame()
    
    for i, item in enumerate(portfolio_candidates):
        pair_str = f"{item['tickers'][0]}-{item['tickers'][1]}"
        model = item['model']
        X_test = item['X_test']
        
        # 1. AI Dự báo trên tập Test
        # Kẹp giá trị dự báo để tránh sai số quá lớn (Clipping)
        preds = model.predict(X_test)
        preds = np.clip(preds, -3.0, 3.0) 
        
        # Đánh giá độ chính xác (R2, RMSE)
        model.evaluate(item['y_test'], preds, pair_name=pair_str)
        
        # 2. Tái tạo DataFrame cho Backtest
        # Lấy lại phần dữ liệu gốc tương ứng với X_test
        test_index = X_test.index
        df_backtest = item['data'].loc[test_index].copy()
        df_backtest['Spread_Z_Forecast'] = preds # Gán dự báo vào
        
        # 3. Sinh Tín hiệu (Signal Logic cũ)
        # Logic: Mua khi AI báo Z < -1, Bán khi AI báo Z > 1
        df_signals = sig_gen.generate_signals(df_backtest, col_name='Spread_Z_Forecast')
        
        # 4. Tính PnL (Backtester cũ)
        original_spread = df_backtest['Spread']
        df_result = backtester.calculate_pnl(df_signals, original_spread)
        
        # Lưu kết quả PnL của cặp này
        weight = final_weights[i] # Tỷ trọng vốn được phân bổ
        
        # Tính lãi/lỗ đóng góp vào Portfolio tổng
        # (Lãi của cặp * Tỷ trọng vốn)
        # Giả sử vốn 1 tỷ, cặp này được 30%, thì lãi tính trên 300tr
        portfolio_daily_returns[pair_str] = df_result['Strategy_PnL'] * weight

        # Hiển thị biểu đồ cho cặp này (Tùy chọn, có thể comment lại nếu nhiều cặp quá)
        # print(f"   -> Vẽ biểu đồ cho {pair_str}...")
        # visualizer.plot_performance(df_result)

    # --------------------------------------------------------------------------
    # BƯỚC 7: TỔNG HỢP KẾT QUẢ DANH MỤC (PORTFOLIO RESULT)
    # --------------------------------------------------------------------------
    print("\n" + "="*70)
    print(" 📊 KẾT QUẢ ĐẦU TƯ CỦA TOÀN BỘ DANH MỤC (PORTFOLIO)")
    print("="*70)
    
    # 1. Tổng hợp PnL hàng ngày của tất cả các cặp
    portfolio_daily_returns['Total_PnL'] = portfolio_daily_returns.sum(axis=1)
    portfolio_daily_returns['Cumulative_PnL'] = portfolio_daily_returns['Total_PnL'].cumsum()
    
    # 2. Tính các chỉ số tài chính
    total_spread_points = portfolio_daily_returns['Cumulative_PnL'].iloc[-1]
    
    # Giả định quy đổi: 1 điểm Spread ~ 1.000 VND (tùy quy ước)
    # Và Vốn đầu tư giả định là 100,000,000 VND để tính %
    # Ở đây ta dùng log return của spread làm % xấp xỉ
    
    # Sharpe Ratio Portfolio
    daily_ret = portfolio_daily_returns['Total_PnL']
    if daily_ret.std() != 0:
        sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252)
    else:
        sharpe = 0
        
    print(f" 💰 Tổng Lãi/Lỗ (Points): {total_spread_points:.4f}")
    print(f" 📈 Sharpe Ratio (Portfolio): {sharpe:.2f}")
    print(f" ⚖️  Tỷ trọng phân bổ: {[round(w*100, 1) for w in final_weights]}%")
    
    if total_spread_points > 0:
        print(" ✅ KẾT LUẬN: Hệ thống có lãi ròng trên tập kiểm thử.")
    else:
        print(" 🔻 KẾT LUẬN: Hệ thống đang lỗ, cần điều chỉnh lại tham số.")

    # 3. Vẽ biểu đồ Tổng tài sản (Portfolio Equity Curve)
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_daily_returns.index, portfolio_daily_returns['Cumulative_PnL'], 
             label='Tổng Tài Sản (Portfolio)', color='purple', linewidth=2)
    plt.fill_between(portfolio_daily_returns.index, portfolio_daily_returns['Cumulative_PnL'], 
                     color='purple', alpha=0.1)
    plt.title(f"Tăng trưởng Tài sản Danh mục (Sharpe: {sharpe:.2f})", fontsize=14)
    plt.ylabel("Lợi nhuận tích lũy (Spread Points)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_advanced_system()