# nhập và tải dữ liệu 
import yfinance as yf
import os # làm việc với thư mục, file
import pandas as pd
class StockData(object):
    # lớp khởi tạo 
    def __init__(self, ticker, start, end):
        self.ticker = ticker # ticker: lấy cổ phiếu của công ty nào
        self.start = start # start: thời điểm bắt đầu
        self.end = end # end: thời điểm kết thúc
        self.data = None # dữ liệu tải về
        self.info = None # thông tin công ty
        self.financials = None # báo cáo tài chính 
    # lấy dữ liệu từ yfinance
    def fetch(self):
        self.data = yf.download(self.ticker, start=self.start, end=self.end, auto_adjust=False)
        return self.data
    # lấy thông tin của công ty
    def get_basic_info(self):
        # lấy thông tin cơ bản của công ty 
        try: 
            ticker_info = yf.Ticker(self.ticker)
            self.info = ticker_info.info
        except Exception as e:
            print("Lỗi lấy dữ liệu thông tin của công ty")
            self.info = {}
        basic_info = {
            "Ten_cong_ty": self.info.get("shortName"), # tên công ty
            "Nganh": self.info.get("industry"), # tên ngành 
            "PE": self.info.get("trailingPE"), # đánh giá mỗi quan hệ giữa giá thị trường và thu nhập trên 1 cổ phiếu
            "PB": self.info.get("priceToBook"), # đánh giá khả năng đầu tư 
            "EPS": self.info.get("trailingEps"), # đánh giá mức độ tăng trưởng của công ty
            "ROE": self.info.get("returnOnEquity") # đánh giá mức độ sử dụng hiệu quả nguồn vốn của công ty
        }
        return basic_info
    # lấy báo cáo tài chính
    def get_financials(self):
        try:
            ticker_financial = yf.Ticker(self.ticker)
            self.financials = {
                "income_statement": ticker_financial.financials, # báo cáo kết quả hoạt động kinh doanh 
                "balance_sheet": ticker_financial.balance_sheet, # bảng cân đối kế toán 
                "cashflow": ticker_financial.cashflow # báo cáo lưu chuyển tiền tệ 
            }
            return self.financials
        except Exception as e:
            print("Lỗi lấy dữ liệu báo cáo tài chính")
            return None 
    # nhập dữ liệu từ file csv
    def load_data(self, path):
        # lấy phần mở rộng của file
        ext = os.path.splitext(path)[-1].lower()
        if ext == ".csv":
            self.data = pd.read_csv(path)
        elif ext in ['.xlsx', 'xls']:
            self.data = pd.read_excel(path)
        else:
            raise ValueError("Tạm thời chỉ hỗ trợ định dạng file đuôi csv, xlsx, xls")
    # lấy dữ liệu
    def get_data(self):
        if self.data is None:
            raise ValueError("Chưa có dữ liệu, vui lòng gọi hàm để lấy dữ liệu")
        return self.data