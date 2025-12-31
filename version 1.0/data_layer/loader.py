import yfinance as yf
import pandas as pd
import os

class DataLoader:
    """lớp tải dữ liệu/ nhập dữ liệu"""
    def __init__(self, start_date=None, end_date=None):
        """
        Khởi tạo DataLoader.
        :param start_date: 'YYYY-MM-DD'
        :param end_date: 'YYYY-MM-DD'
        """
        self.start = start_date # ngày bắt đầu 
        self.end = end_date # ngày kết thúc
        self.tickers_data = {} # dict: key là tên cổ phiếu, value là dataFrame
       
    def download_data(self, tickers: list):
        print(f"Đang tải dữ liệu: {tickers}...")

        # Chuyển string đơn thành list nếu cần
        if isinstance(tickers, str):
            tickers = [tickers]

        # duyệt tải từng cổ phiếu
        for t in tickers:
            try:
                # auto_adjust=False: Giữ nguyên giá Close và Adj Close gốc
                df = yf.download(t, start=self.start, end=self.end, auto_adjust=False, progress=False)
                
                if df.empty:
                    print(f"Không có dữ liệu: {t}")
                    continue
                
                # Đảm bảo index là Datetime chuẩn
                df.index = pd.to_datetime(df.index)
                
                # Nếu yfinance trả về MultiIndex (VD: Price, Ticker) thì biến đổi thành index đơn
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                # cập nhật cổ phiếu
                self.tickers_data[t] = df
                print(f"Đã tải {t}: {len(df)} dòng.")
            except Exception as e:
                print(f"Lỗi tải {t}: {str(e)}")
        
        return self.tickers_data

    def load_data(self, ticker, file_path):
        """đọc dữ liệu file từ máy, hỗ trợ đuôi csv, xlxs, xls"""
        # lấy phần mở rộng của đường dẫn 
        ext = os.path.splitext(file_path)[-1].lower() 
        try:
            if ext == ".csv":
                df = pd.read_csv(file_path, parse_dates=True, index_col=0)
            elif ext in ['.xlsx', 'xls']:
                df = pd.read_excel(file_path, parse_dates=True, index_col=0)
            else:
                raise ValueError("Unsupported format")
            self.tickers_data[ticker] = df
        except Exception as e:
            print(f"Error: {e}")

    def get_fundamental_info(self, ticker):
        """lấy thông tin của cổ phiếu"""
        try:
            t = yf.Ticker(ticker)
            return t.info
        except:
            return {}