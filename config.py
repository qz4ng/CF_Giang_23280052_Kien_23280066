# config.py

#------
# 1. CẤU HÌNH DỮ LIỆU (DATA INPUT)


# Danh sách các mã cổ phiếu muốn theo dõi/giao dịch
# Hệ thống sẽ quét trong nhóm này để tìm ra cặp đôi "bài trùng" (Pair Trading)
TICKERS = [
    "VCB.VN", "BID.VN", "CTG.VN", "ACB.VN",  # Nhóm Ngân hàng
    "VIC.VN", "VHM.VN", "VRE.VN",            # Nhóm Vingroup
    "HPG.VN", "HSG.VN", "NKG.VN",            # Nhóm Thép
    "FPT.VN", "MWG.VN", "MSN.VN",            # Bluechips khác
    "VNM.VN", "GAS.VN", "PLX.VN"
]

# Thời gian lấy dữ liệu lịch sử
START_DATE = '2020-01-01'
END_DATE   = '2023-12-31'

#------
# 2. tham số xử lí (PRE-PROCESSING)
#------

# Ngưỡng để xác định giá trị ngoại lai (Nhiễu)
# Nếu giá biến động > 3 lần độ lệch chuẩn -> Coi là nhiễu và kẹp lại (Winsorize)
OUTLIER_THRESH = 3.0 

#------
# 3. THAM SỐ CHỈ BÁO KỸ THUẬT (FEATURES)
#------

# Cửa sổ quan sát chung (số ngày nhìn lại quá khứ để tính trung bình)
WINDOW_SIZE = 20  

# RSI (Chỉ số sức mạnh tương đối)
RSI_WINDOW = 14   # Chuẩn quốc tế là 14 ngày

# MACD (Chỉ báo dao động)
MACD_FAST   = 12  # Đường nhanh (trung bình 12 ngày)
MACD_SLOW   = 26  # Đường chậm (trung bình 26 ngày)
MACD_SIGNAL = 9   # Đường tín hiệu

# Bollinger Bands (Dải băng Bollinger)
BB_STD_DEV  = 2.0 # Độ rộng của dải băng (2 lần độ lệch chuẩn)

# Pair Trading (Giao dịch cặp)
# Ngưỡng P-value để chấp nhận 2 mã là "đồng tích hợp" (đi cùng nhau)
# Giá trị càng nhỏ (0.01, 0.05) thì mối liên kết càng chặt chẽ.
COINT_PVALUE_THRESH = 0.05 

#------
# 4. THAM SỐ CHIẾN LƯỢC & MÔ HÌNH (STRATEGY & MODEL)
#------

# Tỷ lệ chia tập dữ liệu: 80% để Học (Train), 20% để Kiểm tra (Test)
TRAIN_SPLIT = 0.8 

# --- CHIẾN THUẬT MEAN REVERSION (ĐẢO CHIỀU VỀ TRUNG BÌNH) ---

# Ngưỡng Z-score để VÀO LỆNH (Entry)
# Khi chênh lệch giá (Spread) lệch quá 2.0 đơn vị chuẩn -> Mua/Bán
Z_ENTRY_THRESHOLD = 2.0     

# Ngưỡng Z-score để THOÁT LỆNH (Exit)
# Khi chênh lệch quay về mức 0 (hoặc gần 0) -> Chốt lời
Z_EXIT_THRESHOLD  = 0.0     

# Ngưỡng Cắt lỗ (Stop Loss)
# Nếu lệch quá xa (VD: 3.5), chứng tỏ nhận định sai -> Cắt lỗ ngay
Z_STOP_LOSS = 3.5