# debug.py
import time

class SearchDebugger:
    def __init__(self, config_weights, player_id: int):
        """Khởi tạo bộ Debug độc lập bám theo cấu hình trọng số và ID của Bot"""
        self.weights = config_weights
        self.player_id = player_id

    def format_score(self, raw_score: int, current_depth: int) -> str:
        """Chuẩn hóa điểm số tuyệt đối theo góc nhìn của Player 0: +M{plies}, -M{plies} hoặc [-15, 15]"""
        # Xác định điểm tuyệt đối dưới góc nhìn của Player 0 (Đi trước là +, đi sau là -)
        abs_score = raw_score if self.player_id == 0 else -raw_score
        
        # 1. Kiểm tra trạng thái Sát cục (Mate) thực sự dựa trên WIN_BASE (9000000)
        if abs(raw_score) > 9000000:
            remaining_depth = 10000000 - abs(raw_score)
            plies = current_depth - remaining_depth
            plies = max(1, plies)  # Đảm bảo khoảng cách tối thiểu là 1 nước
            
            if abs_score > 0:
                return f"+M{plies}"
            else:
                return f"-M{plies}"
        
        # 2. Chuẩn hóa điểm CNN Heuristic thông thường về dải [-15, 15]
        norm_score = abs_score / 3000.0
        
        # Khống chế cứng biên độ không vượt quá khung [-15, 15], tuyệt đối không hiện chữ M nếu chưa mate
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
        """Chỉ hiển thị tối đa 3 biến thể xuất sắc nhất tại độ sâu hoàn thiện lớn nhất"""
        nps = int(node_count / elapsed) if elapsed > 0 else node_count
        print(f"  => Kết quả Depth {deepest_depth:02d} ({elapsed:.3f}s) | NPS: {nps:,}")
        
        # top_variations chứa danh sách các tuple dạng: (cột, raw_score, chuỗi_pv)
        for i, (col, raw_score, pv_string) in enumerate(top_variations[:3], 1):
            formatted_val = self.format_score(raw_score, deepest_depth)
            print(f"     [Top {i}] Cột {col} ({formatted_val}) | Chuỗi biến thể: {pv_string}")