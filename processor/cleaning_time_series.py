from statsmodels.tsa.stattools import adfuller # thư viện để kiểm tra tính dừng
import numpy as np
# xử lý sau khi loại bỏ các giá trị null
class CleaningTimeSeries:
    '''
    kiểm tra tính dừng của dữ liệu bằng thống kê
    nếu dữ liệu không có tính dừng thì sử dụng phương pháp để biến đổi dữ liệu
    '''
    # xử lý đặc tính chuỗi thời gian: Tính dừng & Log Return
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
    # def transform_data_to_stationarity(self, column_name):
    #     if not self.check_stationarity(column_name): # nếu như chuỗi không dừng 
    #         # dùng log return 
    #         log_return = np.log(self.data[column_name] / self.data[column_name].shift(1)).dropna()
    #         self.data["Log Return"] = log_return
    #         return log_return
    #     else:
    #         print(f"Chuỗi đã dừng, không cần chuẩn hóa")
    #         return self.data[column_name]
    def transform_data_to_stationarity(self, column_name):
        '''
        Sửa : Tự động tính Log Return nếu chuỗi chưa dừng
        và trả về DataFrame đã cập nhật.
        '''
        if not self.check_stationarity(column_name):
            print(f"- đổi sang Log Return của {column_name}...")
            
            # Tính Log Return: ln(Pt / Pt-1)
            # Lưu vào cột mới để không mất dữ liệu gốc
            self.data["Log_Return"] = np.log(self.data[column_name] / self.data[column_name].shift(1))
            
            # Drop NaN dòng đầu tiên do shift
            self.data = self.data.dropna()
            
            # Kiểm tra lại (Optional)
            self.check_stationarity("Log_Return")
            
            return self.data
        else:
            print("-> Dữ liệu gốc đã dừng, không cần Log transform.")
            return self.data