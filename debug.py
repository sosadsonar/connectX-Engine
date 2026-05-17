# debug.py
import time

class SearchDebugger:
    def __init__(self, config_weights, player_id: int):
        """Khởi tạo bộ Debug độc lập bám theo cấu hình trọng số và ID của Bot"""
        self.weights = config_weights
        self.player_id = player_id

    def format_score(self, raw_score: int) -> str:
        """Chuẩn hóa điểm số tuyệt đối theo góc nhìn của Player 0: +M{plies}, -M{plies} hoặc [-15, 15]"""
        # Xác định điểm tuyệt đối dưới góc nhìn của Player 0 (X)
        abs_score = raw_score if self.player_id == 0 else -raw_score
        
        # 1. Kiểm tra trạng thái Sát cục thực sự dựa trên dải điểm WIN_BASE (9000000)
        if abs(raw_score) > 9000000:
            # Tính toán số nước đi (plies) thực tế còn lại cho đến khi kết thúc ván đấu
            plies = 10000000 - abs(raw_score)
            
            if abs_score > 0:
                return f"+M{plies}"
            else:
                return f"-M{plies}"
        
        # 2. Chuẩn hóa điểm Heuristic thông thường về dải giới hạn cứng [-15, 15]
        norm_score = abs_score / 3000.0
        
        # Khống chế cứng biên độ, tuyệt đối không hiện ký tự M bừa bãi khi chưa sát cục
        if norm_score > 15.0:
            return "+15.00"
        if norm_score < -15.0:
            return "-15.00"
            
        return f"{norm_score:+.2f}"

    def print_start_turn(self, board) -> None:
        """Tính toán Turn Count và hiển thị tiêu đề lượt đi"""
        move_count = bin(board.boards[0] | board.boards[1]).count('1') + 1
        role = "Người đi trước X" if self.player_id == 0 else "Người đi sau O"
        print(f"\n[TURN {move_count:02d}] BOT THINKING (Phe: {role})")

    def print_top_variations(self, deepest_depth: int, elapsed: float, node_count: int, top_variations: list) -> None:
        """Chỉ hiển thị duy nhất 1 lần kết quả Top 3 biến thể PV tinh anh tại tầng hoàn thiện sâu nhất"""
        nps = int(node_count / elapsed) if elapsed > 0 else node_count
        print(f"  => Kết quả Depth {deepest_depth:02d} ({elapsed:.3f}s) | NPS: {nps:,}")
        
        for i, (col, raw_score, pv_string) in enumerate(top_variations[:3], 1):
            formatted_val = self.format_score(raw_score)
            print(f"     [Top {i}] Cột {col} ({formatted_val}) | Chuỗi biến thể: {pv_string}")