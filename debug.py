# debug.py
import time
import math

class SearchDebugger:
    def __init__(self, config_weights, player_id: int):
        """Khởi tạo bộ Debug độc lập bám theo cấu hình trọng số và ID của Bot"""
        self.weights = config_weights
        self.player_id = player_id

    def format_score(self, raw_score: int) -> str:
        """Chuẩn hóa điểm số tuyệt đối theo góc nhìn của Player 0: +M{plies}, -M{plies} hoặc dải nén mịn"""
        # Xác định điểm tuyệt đối dưới góc nhìn của Player 0 (X)
        abs_score = raw_score if self.player_id == 0 else -raw_score
        
        # 1. Kiểm tra trạng thái Sát cục thực sự dựa trên dải điểm WIN_BASE (9000000)
        if abs(raw_score) > 9000000:
            plies = 10000000 - abs(raw_score)
            if abs_score > 0:
                return f"+M{plies}"
            else:
                return f"-M{plies}"
        
        # 2. BỘ LỌC MỊN PHI TUYẾN TÍNH (Non-linear Evaluation Smoothing):
        # Nâng ước số lên tỉ lệ thuận với FORK_SCORE (tương đương 1/10 Fork)
        # giúp điểm số biến động mượt mà, phản ánh đúng bản chất vị trí.
        fork_scale = float(self.weights.get("FORK_SCORE", 700000))
        divisor = max(30000.0, fork_scale / 10.0) # Đặt 1 đòn Fork tương đương 10 điểm lợi thế
        
        val = abs_score / divisor
        
        # Khống chế dải điểm Heuristic biến thiên mịn màng, không bị nhảy vọt bừa bãi
        if val > 0:
            norm_score = min(12.0, val)
        else:
            norm_score = max(-12.0, val)
            
        return f"{norm_score:+.2f}"

    def print_start_turn(self, board) -> None:
        move_count = bin(board.boards[0] | board.boards[1]).count('1') + 1
        role = "Người đi trước X" if self.player_id == 0 else "Người đi sau O"
        print(f"\n[TURN {move_count:02d}] BOT THINKING (Phe: {role})")

    def print_top_variations(self, deepest_depth: int, elapsed: float, node_count: int, top_variations: list) -> None:
        nps = int(node_count / elapsed) if elapsed > 0 else node_count
        print(f"  => Kết quả Depth {deepest_depth:02d} ({elapsed:.3f}s) | NPS: {nps:,}")
        
        for i, (col, raw_score, pv_string) in enumerate(top_variations[:3], 1):
            formatted_val = self.format_score(raw_score)
            print(f"     [Top {i}] Cột {col} ({formatted_val}) | Chuỗi biến thể: {pv_string}")