import pandas as pd
import numpy as np
import scipy.stats as stats
class CleaningBasic:
    '''
    Docstring for CleaningBasic:
    dùng để tiền xử lý dữ liệu:
    - chuyển đổi multiIndex về dạng đơn
    - xử lý thời gian 
    - missing values
    - outliers (giảm ảnh hưởng không loại bỏ ngày) 
    '''
    def __init__(self, data: pd.DataFrame):
        '''hàm khởi tạo'''
        self.data = data.copy() # không ảnh hưởng đến dữ liệu gốc 
    
    def change_index(self):
        '''
        chuyển dữ liệu có dạng multiIndex thành dạng dataFrame chuẩn:
        - lấy tên cổ phiếu từ multiIndex 
        - đưa date thành DateTime index 
        - loại bỏ multiIndex
        '''
        ticker_name = self.data.columns.get_level_values(1)[1] # lấy tên cổ phiếu 
        self.data = self.data.reset_index() # đổi date thành index
        self.data['Ticker'] = ticker_name # thêm cột tên cổ phiếu
        self.data["Date"] = pd.to_datetime(self.data["Date"]) # đảm bảo cột thời gian
        self.data = self.data.set_index('Date').sort_index() # sắp xếp thời gian theo thứ tự tăng dần
        self.data.columns = self.data.columns.get_level_values(0) # loại bỏ multiIndex
        return self.data

    # missing values 
    def fill_missing_values(self, method = 'linear'):
        '''điền khuyêt dữ liệu bằng phương pháp nội suy'''
        self.data = self.data.interpolate(method=method)
        return self.data
   
    def fill_missing_values_ffill_bfill(self):
        '''điền khuyết dữ liệu bằng phương pháp forward fill và back fill'''
        self.data = self.data.ffill().bfill()
        return self.data
    
    # xử lý outlier
    def clip_outlier_iqr(self, column, factor = 1.5):
        '''dùng winsorize để thay thế các giá trị ngoại lai thành các giá trị ít cực đoan hơn + IQR'''
        Q1 = self.data[column].quantile(0.25) # tứ phân vị thứ nhất
        Q3 = self.data[column].quantile(0.75) # tứ phân vị thứ ba
        IQR = Q3 - Q1
        lower = Q1 - factor*IQR # khoảng giá trị ở dưới
        upper = Q3 + factor*IQR # khoảng giá trị ở trên
        self.data[column + '_clipped'] = self.data[column].clip(lower, upper) # tạo cột mới với giá trị đã xử lý
        return self.data
    
    def clip_outlier_zscore(self, column, threshold = 3):
        '''dùng winsorize để thay thế các giá trị ngoại lai thành các giá trị ít cực đoan hơn + Zscore'''
        z_score = stats.zscore(self.data[column])   # vì dữ liệu đã được xử lý missing values
        mask_upper = z_score > threshold    # giá trị quá cao
        mask_lower = z_score < -threshold   # giá trị quá thấp
        col = self.data[column].copy()
        col[mask_lower] = col[~mask_lower].min()    # thay giá trị ngoại lai nhỏ = giá trị nhỏ nhất trong nhóm bình thường
        col[mask_upper] = col[~mask_upper].max()    # thay giá trị ngoại lai lớn = giá trị lớn nhất trong nhóm bình thường
        self.data[column + '_clipped'] = col 
        return self.data
    
    # chuẩn hóa dữ liệu (không loại bỏ dữ liệu mà chỉ đưa về cùng thang đo)
    def IQR_scaling(self, column):  
        '''chuẩn hóa dữ liệu bằng IQR'''
        Q1 = self.data[column].quantile(0.25)
        Q3 = self.data[column].quantile(0.75)
        IQR = Q3 - Q1
        self.data[column + '_scaled_IQR'] = (self.data[column] - Q1)/ IQR
        return self.data
    
    def log_transform(self, column):
        '''chuẩn hóa dữ liệu bằng phương pháp log'''
        self.data[column + '_scaled_log'] = np.log1p(self.data[column])
        return self.data

    def MinMax_Scale(self, column):
        '''chuẩn hóa dữ liệu bằng phương pháp min max'''
        min_val = self.data[column].min()
        max_val = self.data[column].max()
        self.data[column + '_scaled_minmax'] = (self.data[column] - min_val)/ (max_val - min_val)
        return self.data