import pandas as pd
import numpy as np

class Backtester:
    """
    Tính toán hiệu suất giao dịch (PnL - Profit and Loss).
    """
    def __init__(self):
        pass

    def calculate_pnl(self, df, original_spread):
        """
        Tính lãi/lỗ dựa trên tín hiệu và thay đổi của Spread.
        Input:
            df: DataFrame đã có cột 'Position' (Vị thế).
            original_spread: Series giá trị Spread gốc ($A - beta*$B).
        """
        df = df.copy()
        
        # Tính thay đổi giá của Spread hôm nay so với hôm qua
        # Spread tăng hay giảm?
        df['Spread_Change'] = original_spread - original_spread.shift(1)
        
        # Tính Lợi nhuận chiến lược
        # Lợi nhuận = Vị thế hôm qua * Mức thay đổi giá hôm nay
        # Nếu đang Long (1) và giá tăng (+) -> Lãi (+)
        # Nếu đang Short (-1) và giá giảm (-) -> Lãi (+) (Âm nhân Âm ra Dương)
        # Nếu đang Short (-1) và giá tăng (+) -> Lỗ (-)
        df['Strategy_PnL'] = df['Position'] * df['Spread_Change']
        
        # Tính tổng lãi lỗ tích lũy (Cumulative PnL)
        # Để vẽ biểu đồ tài sản tăng trưởng thế nào
        df['Cumulative_PnL'] = df['Strategy_PnL'].cumsum()
        
        return df

    def evaluate_performance(self, df):
        """
        Đánh giá các chỉ số tài chính: Sharpe Ratio, Win Rate...
        """
        # Tổng lãi/lỗ cuối cùng
        total_profit = df['Cumulative_PnL'].iloc[-1]
        
        # Số lượng lệnh (Số lần đổi trạng thái từ 0 sang khác 0)
        trades = df['Position'].diff().fillna(0).abs()
        num_trades = trades[trades > 0].count()
        
        # Sharpe Ratio (Lợi nhuận / Rủi ro)
        # Giả sử risk-free rate = 0 cho đơn giản
        daily_returns = df['Strategy_PnL']
        if daily_returns.std() == 0:
            sharpe_ratio = 0
        else:
            # Nhân căn(252) để quy đổi ra năm (năm có 252 ngày giao dịch)
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        
        print(f"KẾT QUẢ BACKTEST:")
        print(f"- Tổng Lãi/Lỗ: {total_profit:.4f} điểm Spread")
        print(f"- Số lần giao dịch: {num_trades}")
        print(f"- Sharpe Ratio: {sharpe_ratio:.2f}")
        
        if total_profit > 0:
            print("Chiến thuật CÓ LỜI")
        else:
            print("Chiến thuật THUA LỖ")
            
        return total_profit, sharpe_ratio