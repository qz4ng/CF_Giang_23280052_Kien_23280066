import pandas as pd
import numpy as np
import scipy.stats as stats
class CleaningBasic:
    # khởi tạo
    def __init__(self, data: pd.DataFrame):
        self.data = data
    # missing values 
    # điền khuyết dữ liệu bằng phương pháp nội suy
    def fill_missing_values(self, method = 'linear'):
        self.data = self.data.interpolate(method=method)
        return self.data
    # điền khuyết bằng phương pháp forward fill and back fill
    def fill_missing_values_ffill_bfill(self):
        self.data = self.data.fillna(method = 'ffill').fillna(method = 'bfill')
        return self.data
    # xử lý outlier
    def IQR(self, column, factor = 1.5):
        # loại bỏ giá trị ngoại lai bằng phương pháp IQR
        Q1 = self.data[column].quantile(0.25) # tứ phân vị thứ nhất
        Q3 = self.data[column].quantile(0.75) # tứ phân vị thứ ba
        IQR = Q3 - Q1
        lower = Q1 - factor*IQR # khoảng giá trị ở dưới
        upper = Q3 + factor*IQR # khoảng giá trị ở trên
        self.data = self.data[(self.data[column] >= lower) & (self.data[column] <= upper)]
        return self.data
    def Zscore(self, column, threshold = 3):
        # loại bỏ giá trị ngoại lai bằng phương pháp zscore
        z_score = np.abs(stats.zscore(self.data[column])) # vì dữ liệu đã được xử lý missing values
        self.data = self.data.loc[z_score < threshold] # lọc những giá trị trong khoảng zscore
        return self.data
    # chuẩn hóa dữ liệu (không loại bỏ dữ liệu mà chỉ đưa về cùng thang đo)
    def IQR_scaling(self, column):  
        Q1 = self.data[column].quantile(0.25)
        Q3 = self.data[column].quantile(0.75)
        IQR = Q3 - Q1
        self.data[column + '_scaled_IQR'] = (self.data[column] - Q1)/ IQR
        return self.data
    # chuẩn hóa dữ liệu dựa vào log
    def log_transform(self, column):
        self.data[column + '_scaled_log'] = np.log1p(self.data[column])
        return self.data
    # chuẩn hóa dữ liệu dựa vào phương pháp min max
    def MinMax_Scale(self, column):
        min_val = self.data[column].min()
        max_val = self.data[column].max()
        self.data[column + '_scaled_minmax'] = (self.data[column] - min_val)/ (max_val - min_val)
        return self.data

