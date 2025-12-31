import numpy as np
import pandas as pd
from scipy.optimize import minimize

class PortfolioOptimizer:
    """
    [MATH CORE]
    Giải bài toán tối ưu hóa Mean-Variance (Markowitz).
    Mục tiêu: Tìm bộ tỷ trọng (Weights) để Tối đa hóa Sharpe Ratio.
    """
    def __init__(self, risk_free_rate=0.0):
        self.risk_free_rate = risk_free_rate

    def optimize_sharpe_ratio(self, expected_returns, cov_matrix):
        """
        Input:
            expected_returns: Mảng 1D lợi nhuận kỳ vọng (VD: [0.1, 0.5, 0.2])
            cov_matrix: Ma trận hiệp phương sai (Rủi ro & Tương quan)
        Output:
            optimal_weights: Mảng tỷ trọng (VD: [0.1, 0.6, 0.3])
        """
        n_assets = len(expected_returns)
        
        # 1. Khởi tạo: Chia đều tiền (Equal Weights)
        init_guess = n_assets * [1. / n_assets,]
        
        # 2. Ràng buộc (Constraints): Tổng tỷ trọng phải bằng 1 (100% vốn)
        # 'eq' nghĩa là equality (phương trình bằng 0) -> sum(x) - 1 = 0
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # 3. Giới hạn (Bounds): Mỗi mã từ 0% đến 100% (Long-only)
        # Nếu muốn tránh dồn trứng 1 giỏ, có thể chỉnh (0.0, 0.4) -> Max 40%
        bounds = tuple((0.0, 1.0) for asset in range(n_assets))
        
        # 4. Arguments truyền vào hàm mục tiêu
        args = (expected_returns, cov_matrix, self.risk_free_rate)
        
        # 5. Chạy thuật toán tối ưu SLSQP
        # (Sequential Least Squares Programming - Chuyên trị bài toán phi tuyến tính có ràng buộc)
        try:
            result = minimize(self._negative_sharpe, init_guess, args=args,
                              method='SLSQP', bounds=bounds, constraints=constraints)
            
            if not result.success:
                print(f"   [WARN] Tối ưu hóa thất bại: {result.message}. Dùng tỷ trọng đều.")
                return np.array(init_guess)
                
            return result.x
            
        except Exception as e:
            print(f"   [ERROR] Lỗi Optimizer: {str(e)}. Dùng tỷ trọng đều.")
            return np.array(init_guess)

    def _negative_sharpe(self, weights, expected_returns, cov_matrix, risk_free_rate):
        """
        Hàm mục tiêu (Objective Function).
        Scipy chỉ có hàm minimize (tìm cực tiểu), nên ta minimize ÂM Sharpe 
        (Tương đương với maximize DƯƠNG Sharpe).
        """
        p_ret, p_vol = self._get_portfolio_metrics(weights, expected_returns, cov_matrix)
        
        # Tránh chia cho 0 hoặc rủi ro quá thấp
        if p_vol < 1e-6: 
            return 0
        
        sharpe = (p_ret - risk_free_rate) / p_vol
        return -sharpe

    def _get_portfolio_metrics(self, weights, expected_returns, cov_matrix):
        """
        Tính Lợi nhuận và Rủi ro của Portfolio với bộ weight hiện tại.
        Công thức đại số tuyến tính:
        - Return = w * R
        - Risk = sqrt(w.T * Cov * w)
        """
        port_ret = np.sum(expected_returns * weights)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return port_ret, port_vol