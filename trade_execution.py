import pandas as pd
import numpy as np
import pickle
import datetime
import config_updated as config

# Import các công cụ cần thiết để tính feature
from data_layer.loader import DataLoader
from data_layer.processor import DataProcessor
from feature_layer.trend import TrendIndicators
from feature_layer.momentum import MomentumIndicators
from feature_layer.volatility import VolatilityIndicators
from model_layer.data_handler_updated import DataHandlerUpdated

# Import Allocator để chia tiền
from portfolio_layer.allocator import StrategyAllocator

MODEL_PATH = 'models/current_portfolio_state.pkl'

def run_daily_trading():
    print(f" GIAO DỊCH HÀNG NGÀY - {datetime.date.today()}")
    print("="*60)

    # 1. KIỂM TRA MODEL
    try:
        with open(MODEL_PATH, 'rb') as f:
            portfolio_state = pickle.load(f)
        print(f" (gồm {len(portfolio_state)} cặp).")
    except FileNotFoundError:
        print("LỖI: Không tìm thấy file model. ")
        return

    # 2. TẢI DỮ LIỆU MỚI NHẤT (REAL-TIME DATA)
    # Lấy dữ liệu đến ngày hôm nay để tính chỉ báo
    print("\n[1/3] cập nhật dữ liệu  mới nhất...")
    today_str = datetime.date.today().strftime('%Y-%m-%d')
    
    # Cần lấy dư ra khoảng 100 ngày trước đó để tính đủ MA, RSI, Rolling
    start_lookback = (datetime.date.today() - datetime.timedelta(days=150)).strftime('%Y-%m-%d')
    
    loader = DataLoader(start_lookback, today_str)
    # Chỉ tải những mã có trong portfolio đã chọn
    active_tickers = set()
    for item in portfolio_state:
        active_tickers.add(item['tickers'][0])
        active_tickers.add(item['tickers'][1])
    
    raw_data = loader.download_data(list(active_tickers))
    processor = DataProcessor()
    processed_data = processor.process_all(raw_data)

    # 3. CHUẨN BỊ DỮ LIỆU ĐỂ DỰ BÁO
    print("\n[2/3] Dự báo tín hiệu...")
    
    pairs_ready_to_trade = []
    
    for item in portfolio_state:
        t1, t2 = item['tickers']
        
        # Kiểm tra xem có đủ dữ liệu hôm nay không
        if t1 not in processed_data or t2 not in processed_data:
            print(f"  Thiếu dữ liệu cho cặp {t1}-{t2}. Bỏ qua.")
            continue
            
        df1 = processed_data[t1]
        df2 = processed_data[t2]
        
        # --- A. TÍNH INDICATORS (Phải khớp logic lúc train) ---
        trend = TrendIndicators(); mom = MomentumIndicators(); vol = VolatilityIndicators()
        for df in [df1, df2]:
            df = trend.add_macd(df)
            df = trend.add_sma_distance(df)
            df = mom.add_rsi(df)
            df = vol.add_bollinger_bands(df)
            
        # --- B. TÍNH SPREAD (Dùng Beta đã lưu trong model) ---
        # Lưu ý: Lúc rebalance ta đã lưu 'beta_avg'. 
        # Nếu muốn chính xác tuyệt đối thì tính lại Rolling Beta tại thời điểm này.
        # Ở đây ta tính lại Rolling Beta để khớp thời gian thực
        from feature_layer.pairs_updated import PairsIndicatorsUpdated
        pairs_logic = PairsIndicatorsUpdated()
        
        # Tính Rolling Spread
        df_pair, _ = pairs_logic.calculate_rolling_spread(
            df1, df2, window=getattr(config, 'ROLLING_WINDOW', 60)
        )
        
        # Cập nhật lại dữ liệu mới nhất vào item
        item['data'] = df_pair 
        
        # --- C. CHUẨN BỊ INPUT CHO AI ---
        handler = DataHandlerUpdated()
        # Tạo Lags, Rolling stats...
        X_full, _ = handler.create_dataset(df_pair, target_col='Spread_Z', lags=getattr(config, 'LAG_DAYS', 3))
        
        # Quan trọng: Cần gán feature_cols đã lưu để đảm bảo thứ tự cột đúng
        handler.feature_cols = item['feature_cols']
        
        # Đưa vào list để Allocator xử lý
        # Lưu ý: Allocator cần 'handler' có chứa 'scaler' ĐÃ FIT.
        # Scaler đã được lưu trong item['scaler'] lúc rebalance
        handler.scaler = item['scaler'] 
        item['handler'] = handler
        
        pairs_ready_to_trade.append(item)

    # 4. PHÂN BỔ VỐN & RA QUYẾT ĐỊNH
    print("\n[3/3] Tối ưu hóa danh mục & Ra phiếu lệnh")
    
    allocator = StrategyAllocator(risk_manager=True)
    weights = allocator.allocate_capital(pairs_ready_to_trade)
    
    # 5. IN PHIẾU LỆNH (ORDER TICKET)
  
    print(f" KHUYẾN NGHỊ GIAO DỊCH NGÀY {datetime.date.today()}")
    print("="*60)
    
    for i, item in enumerate(pairs_ready_to_trade):
        t1, t2 = item['tickers']
        w = weights[i] * 100
        
        # Logic đơn giản: Nếu tỷ trọng > 0 thì xem Long hay Short Spread
        if w > 1.0: # Chỉ hiển thị nếu tỷ trọng đáng kể (>1%)
            # Lấy giá trị Z-score dự báo (đã tính ngầm trong allocator, giờ lấy lại để hiển thị)
            # Hoặc truy cập trực tiếp nếu allocator trả về chi tiết hơn.
            # Ở đây in thông tin phân bổ
            print(f" CẶP: {t1} & {t2}")
            print(f"    - Tỷ trọng vốn: {w:.2f}%")
            print(f"    - Hành động: Rebalance danh mục theo tỷ trọng này.")
            print("-" * 30)
            
    print("="*60)

if __name__ == "__main__":
    run_daily_trading()