import pandas as pd
import config
from data_layer.loader import DataLoader
from data_layer.processor import DataProcessor
from feature_layer.trend import TrendIndicators
from feature_layer.momentum import MomentumIndicators
from feature_layer.volatility import VolatilityIndicators
from feature_layer.pairs import PairsIndicators
from model_layer.data_handler import DataHandler
from model_layer.regressor import LinearTrader
from strategy_layer.signals import SignalLogic
from strategy_layer.backtester import Backtester

def run_system():
    print("\n" + "="*50)
    print(" HỆ THỐNG QUANT TRADING BẮT ĐẦU KHỞI ĐỘNG")
    print("="*50)

    
    # BƯỚC 1: ĐI CHỢ (Tải & Xử lý dữ liệu)
    
    print("\n[1/5] Đang tải và xử lý dữ liệu...")
    
    # 1. Tải dữ liệu thô
    loader = DataLoader(config.START_DATE, config.END_DATE)
    raw_data = loader.download_data(config.TICKERS)
    
    # 2. Làm sạch & Đồng bộ
    processor = DataProcessor()
    processed_data = processor.process_all(raw_data)
    
    print(f"   => Đã chuẩn bị xong dữ liệu sạch cho {len(processed_data)} mã.")

    
    # BƯỚC 2: SƠ CHẾ (Tính toán chỉ báo kỹ thuật)
    
    print("\n[2/5] 🛠️ Đang tính toán chỉ báo (RSI, MACD, Bollinger)...")
    
    trend = TrendIndicators()
    mom = MomentumIndicators()
    vol = VolatilityIndicators()
    
    # Lặp qua từng mã để thêm gia vị (Feature)
    for ticker, df in processed_data.items():
        # Thêm xu hướng
        df = trend.add_sma_distance(df)
        df = trend.add_macd(df)
        # Thêm sức mạnh
        df = mom.add_rsi(df)
        df = mom.add_roc(df)
        # Thêm biến động
        df = vol.add_bollinger_bands(df)
        
        # Lưu ngược lại vào dictionary (bỏ các dòng NaN do tính toán)
        processed_data[ticker] = df.dropna()

    
    # BƯỚC 3: TÌM CẶP ĐÔI HOÀN HẢO (Pair Selection)
    
    print("\n[3/5] Đang quét tìm cặp cổ phiếu đồng tích hợp...")
    
    pairs_logic = PairsIndicators()
    best_pair, p_value = pairs_logic.find_best_pair(processed_data)
    
    if best_pair is None:
        print(" Rất tiếc! Không tìm thấy cặp nào đủ tiêu chuẩn (p-value quá cao).")
        print("   -> Thử nới lỏng COINT_PVALUE_THRESH trong config.py hoặc thêm mã khác.")
        return

    print(f"  CẶP ĐƯỢC CHỌN: {best_pair[0]} & {best_pair[1]}")
    print(f"   - Độ tin cậy (p-value): {p_value:.5f} (Càng nhỏ càng tốt)")
    
    # Tính toán Spread và Z-score cho cặp này
    df_pair, beta = pairs_logic.calculate_spread_zscore(
        processed_data[best_pair[0]], 
        processed_data[best_pair[1]]
    )
    print(f"   - Tỷ lệ Hedge (Beta): {beta:.4f}")

    
    # BƯỚC 4: HUẤN LUYỆN MÔ HÌNH (Training AI)
    
    print("\n[4/5] Đang huấn luyện mô hình dự báo Spread...")
    
    handler = DataHandler()
    
    # Tạo đề bài (X) và đáp án (y)
    # Target là 'Spread_Z' (Dự báo Z-score ngày mai)
    X, y = handler.create_dataset(df_pair, target_col='Spread_Z')
    
    # Chia tập Train/Test
    X_train, X_test, y_train, y_test = handler.split_data(X, y)
    
    # Khởi tạo và dạy Model
    model = LinearTrader()
    model.train(X_train, y_train)
    
    # Đánh giá năng lực học tập
    predictions = model.predict(X_test)
    rmse, r2 = model.evaluate(y_test, predictions)

    
    # BƯỚC 5: CHIẾN THUẬT & BACKTEST (Chạy thử nghiệm)
    
    print("\n[5/5]  Đang chạy Backtest chiến thuật Mean Reversion...")
    
    # Lấy lại DataFrame gốc tương ứng với tập Test để biết ngày tháng
    test_start_index = len(X_train)
    # Cắt lấy đoạn dữ liệu Test (phải khớp độ dài với predictions)
    df_backtest = df_pair.iloc[test_start_index : test_start_index + len(predictions)].copy()
    
    # Gán dự báo của AI vào cột 'Spread_Z' để Strategy ra quyết định
    # (Ở đây ta giả định tin tưởng hoàn toàn vào dự báo của AI)
    df_backtest['Spread_Z_Forecast'] = predictions
    
    # Sinh tín hiệu Mua/Bán dựa trên Z-score DỰ BÁO
    sig_gen = SignalLogic()
    # Ta dùng cột Forecast để ra quyết định
    df_signals = sig_gen.generate_signals(df_backtest.rename(columns={'Spread_Z_Forecast': 'Spread_Z'}))
    
    # Tính toán Lãi/Lỗ
    backtester = Backtester()
    # Cần truyền vào Spread gốc (chưa Z-score) để tính tiền thật
    original_spread_series = df_backtest['Spread']
    
    df_result = backtester.calculate_pnl(df_signals, original_spread_series)
    final_pnl, sharpe = backtester.evaluate_performance(df_result)
    
    print("\n" + "="*10)
    print("Hoàn tất")
    print("="*50)

if __name__ == "__main__":
    run_system()