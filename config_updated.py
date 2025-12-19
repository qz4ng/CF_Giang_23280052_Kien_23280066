# config_updated.py

# 
# 1. CẤU HÌNH DỮ LIỆU (DATA INPUT)
# 

# Danh sách mở rộng đa ngành để thuật toán Clustering hoạt động hiệu quả
TICKERS = [
    #  Nhóm Ngân hàng (Banks) 
    "VCB.VN", "BID.VN", "CTG.VN", "ACB.VN", "MBB.VN", "TCB.VN", "STB.VN",
    #  Nhóm Bất động sản (Real Estate) 
    "VIC.VN", "VHM.VN", "VRE.VN", "NVL.VN", "KDH.VN",
    #  Nhóm Thép (Steel) 
    "HPG.VN", "HSG.VN", "NKG.VN",
    #  Nhóm Chứng khoán (Securities) 
    "SSI.VN", "VND.VN", "VCI.VN",
    #  Nhóm Bán lẻ & Công nghệ (Retail & Tech) 
    "FPT.VN", "MWG.VN", "MSN.VN", "PNJ.VN",
    #  Nhóm Dầu khí & Năng lượng (Energy) 
    "GAS.VN", "PLX.VN", "PVD.VN", "POW.VN"
]

# Thời gian lấy dữ liệu (Nên lấy dài > 3 năm để AI học đủ các chu kỳ)
START_DATE = '2020-01-01'
END_DATE   = '2023-12-31' # Hoặc dùng datetime.date.today() trong code chạy thực

# 
# 2. TIỀN XỬ LÝ (PRE-PROCESSING)
# 
OUTLIER_THRESH = 3.0  # Ngưỡng lọc nhiễu (Winsorize)

# 
# 3. CHỈ BÁO KỸ THUẬT (FEATURE ENGINEERING)
# 
WINDOW_SIZE = 20      # Cửa sổ chung cho SMA, Bollinger
RSI_WINDOW  = 14
MACD_FAST   = 12
MACD_SLOW   = 26
MACD_SIGNAL = 9
BB_STD_DEV  = 2.0

# 
# 4. GIAO DỊCH CẶP & THÍCH NGHI (PAIRS TRADING & ADAPTABILITY)
# 
# Ngưỡng P-value để chấp nhận cặp đồng tích hợp
COINT_PVALUE_THRESH = 0.05 

# [QUAN TRỌNG] Cửa sổ trượt để tính Beta động (Rolling Beta)
# Giúp hệ thống thích nghi khi mối quan hệ giữa 2 mã thay đổi
ROLLING_WINDOW = 60   

# 
# 5. MÔ HÌNH HỌC MÁY (MACHINE LEARNING - RANDOM FOREST)
# 
TRAIN_SPLIT     = 0.8 # 80% Train, 20% Test
RF_N_ESTIMATORS = 200 # Số lượng cây quyết định
RF_MAX_DEPTH    = 5  # Độ sâu tối đa của cây (tránh overfitting)

# [QUAN TRỌNG] Trí nhớ của AI (Memory)
# Số ngày nhìn lại quá khứ để tạo Lag Features (t-1, t-2, t-3)
LAG_DAYS        = 3   

# 
# 6. QUẢN LÝ DANH MỤC & RỦI RO (PORTFOLIO & RISK)
# 
# Số nhóm ngành muốn phân chia (Clustering)
N_CLUSTERS      = 4   

# Lãi suất phi rủi ro (để tính Sharpe Ratio, VD: Lãi suất trái phiếu CP)
RISK_FREE_RATE  = 0.0