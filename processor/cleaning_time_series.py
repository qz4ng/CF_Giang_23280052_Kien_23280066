from statsmodels.tsa.stattools import adfuller # thư viện để kiểm tra tính dừng
import numpy as np
# xử lý sau khi loại bỏ các giá trị null
class CleaningTimeSeries:
    '''
    kiểm tra tính dừng của dữ liệu bằng thống kê
    nếu dữ liệu không có tính dừng thì sử dụng phương pháp để biến đổi dữ liệu
    '''
    def __init__(self, data):
        self.data = data # một dataframe 
    def check_stationarity(self, column_name):
        # colume_name: cột kiểm tra dữ liệu
        result = adfuller(self.data[column_name].dropna()) # mảng với [1]: giá trị thống kê, [2]: p-value
        p_value = result[1]
        print(f"p-value: {p_value}")
        if (p_value < 0.05):
            print("Chuỗi dừng")
            return True
        else:
            print("Chuỗi không dừng")
            return False
    def transform_data_to_stationarity(self, column_name):
        if not self.check_stationarity(column_name): # nếu như chuỗi không dừng 
            # dùng log return 
            log_return = np.log(self.data[column_name] / self.data[column_name].shift(1)).dropna()
            self.data["Log Return"] = log_return
            return log_return
        else:
            print(f"Chuỗi đã dừng, không cần chuẩn hóa")
            return self.data[column_name]