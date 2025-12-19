#dùng thuật toán K-Means để gom nhóm cổ phiếu.
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class MarketCluster:
    """
    Phân cụm cổ phiếu dựa trên hành vi biến động giá (Correlation).
    """
    def __init__(self, n_clusters=4):
        self.n_clusters = n_clusters # Chia làm 4 nhóm ngành chính

    def cluster_stocks(self, data_dict):
        """
        Input: Dictionary chứa Dataframe của từng mã.
        Output: Dictionary {Mã: Label nhóm}, VD: {'VCB': 0, 'BID': 0, 'HPG': 1...}
        """
        # 1. Tạo DataFrame lợi nhuận gộp chung
        # Chỉ lấy cột 'Adj Close' của tất cả các mã
        prices = pd.DataFrame({k: v['Adj Close'] for k, v in data_dict.items()})
        
        # Tính lợi nhuận hàng ngày (Returns)
        returns = prices.pct_change().dropna()
        
        # 2. Xử lý dữ liệu cho K-Means
        # K-Means gom nhóm dựa trên "Tính tương quan". 
        # Chúng ta dùng Transpose (.T) để gom theo Cột (Mã CK) thay vì Dòng (Ngày)
        X = returns.corr().values 
        
        # 3. Chạy K-Means
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        kmeans.fit(X)
        
        labels = kmeans.labels_
        tickers = returns.columns
        
        # 4. Gán nhãn
        cluster_map = {}
        print(f"\n--- KẾT QUẢ PHÂN CỤM (CLUSTERING) ---")
        for i in range(self.n_clusters):
            group = tickers[labels == i].tolist()
            print(f"Nhóm {i}: {group}")
            for t in group:
                cluster_map[t] = i
                
        return cluster_map