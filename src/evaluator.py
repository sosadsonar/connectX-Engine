# evaluator.py
from board import ConnectXBoard

class BitboardCNNEvaluator:
    def __init__(self, config_weights):
        """Bộ lượng giá Bitwise CNN hình học - Bản tối ưu hóa triệt tiêu overhead tính toán tĩnh"""
        self.weights = config_weights
        
        # Cờ đánh dấu để khởi tạo các biến tĩnh theo kích thước bàn cờ thực tế (Lazy Init)
        self.is_initialized = False
        
        # Khởi tạo sẵn các hằng số lũy thừa để giải phóng CPU khỏi phép tính `**`
        self.my_edge_pow = self.weights["K_EDGE"] ** self.weights["CNN_POWER"]
        self.my_corner_pow = self.weights["K_CORNER"] ** self.weights["CNN_POWER"]
        self.my_both_pow = (self.weights["K_EDGE"] + self.weights["K_CORNER"]) ** self.weights["CNN_POWER"]
        
        # Cache các trọng số thường dùng để tăng tốc độ lookup thuộc tính của dict
        self.fork_score = int(self.weights["FORK_SCORE"])
        self.threat_score = self.weights["THREAT_SCORE"]
        self.threat_score_opp = self.weights["THREAT_SCORE"] * 1.2
        self.c_smooth = self.weights["C_SMOOTH"]
        self.max_strat = self.weights["MAX_STRATEGIC"]
        self.alpha_balanced = self.weights["ALPHA_BALANCED"]
        self.alpha_defensive = self.weights["ALPHA_DEFENSIVE"]
        self.win_base = self.weights["WIN_BASE"]

    def _lazy_init(self, board: ConnectXBoard):
        """Khởi tạo một lần duy nhất các mặt nạ dịch bit cố định dựa trên kích thước bàn cờ"""
        self.col_height = board.col_height
        center_col = board.w // 2
        
        # Tính toán sẵn Center Mask và Flank Mask
        self.center_mask = ((1 << board.h) - 1) << (center_col * self.col_height)
        
        self.flank_mask = 0
        if center_col - 1 >= 0:
            self.flank_mask |= ((1 << board.h) - 1) << ((center_col - 1) * self.col_height)
        if center_col + 1 < board.w:
            self.flank_mask |= ((1 << board.h) - 1) << ((center_col + 1) * self.col_height)
            
        self.is_initialized = True

    def evaluate(self, board: ConnectXBoard, current_player_id: int) -> int:
        if not self.is_initialized:
            self._lazy_init(board)

        my_id = current_player_id
        opp_id = 1 - my_id
        
        # --- PHASE 1: GIẢ LẬP 1-STEP THREATS & FORK DETECTION ---
        my_threats = 0
        opp_threats = 0
        my_parity_threats = 0
        opp_parity_threats = 0
        
        valid_cols = board.get_valid_cols()
        
        # Giữ nguyên cấu trúc của bạn nhưng tối ưu thứ tự đọc ghi
        for col in valid_cols:
            board.make_move(col, my_id)
            if board.check_win(my_id):
                my_threats += 1
                row = (board.heights[col] - 1) % self.col_height
                if (my_id == 0 and row % 2 == 0) or (my_id == 1 and row % 2 != 0):
                    my_parity_threats += 1
            board.undo_move(col, my_id)
            
            board.make_move(col, opp_id)
            if board.check_win(opp_id):
                opp_threats += 1
                row = (board.heights[col] - 1) % self.col_height
                if (opp_id == 0 and row % 2 == 0) or (opp_id == 1 and row % 2 != 0):
                    opp_parity_threats += 1
            board.undo_move(col, opp_id)

        if opp_threats >= 2: return -self.fork_score
        if my_threats >= 2: return self.fork_score

        # --- PHASE 2: THƯỞNG ĐIỂM CHIẾM TRỤC TRUNG TÂM (Dùng bit_count siêu tốc) ---
        my_board = board.boards[my_id]
        opp_board = board.boards[opp_id]

        my_center_bits = (my_board & self.center_mask).bit_count()
        opp_center_bits = (opp_board & self.center_mask).bit_count()
        my_flank_bits = (my_board & self.flank_mask).bit_count()
        opp_flank_bits = (opp_board & self.flank_mask).bit_count()

        center_score = (my_center_bits - opp_center_bits) * 150 + (my_flank_bits - opp_flank_bits) * 50

        # --- PHASE 3: VECTOR HÓA CNN FEATURE MAP ---
        empty_mask = (~(my_board | opp_board)) & board.valid_mask
        
        my_edges_mask = 0
        for s in board.shifts[:2]:
            my_edges_mask |= ((my_board << s) & board.valid_mask) | ((my_board >> s) & board.valid_mask)
        my_corners_mask = 0
        for s in board.shifts[2:]:
            my_corners_mask |= ((my_board << s) & board.valid_mask) | ((my_board >> s) & board.valid_mask)
            
        opp_edges_mask = 0
        for s in board.shifts[:2]:
            opp_edges_mask |= ((opp_board << s) & board.valid_mask) | ((opp_board >> s) & board.valid_mask)
        opp_corners_mask = 0
        for s in board.shifts[2:]:
            opp_corners_mask |= ((opp_board << s) & board.valid_mask) | ((opp_board >> s) & board.valid_mask)

        # Trích xuất không gian ô trống bằng toán tử bitwise
        my_edges_mask &= empty_mask
        my_corners_mask &= empty_mask
        opp_edges_mask &= empty_mask
        opp_corners_mask &= empty_mask

        total_my_cnn_score = (
            ((my_edges_mask & ~my_corners_mask).bit_count() * self.my_edge_pow) +
            ((my_corners_mask & ~my_edges_mask).bit_count() * self.my_corner_pow) +
            ((my_edges_mask & my_corners_mask).bit_count() * self.my_both_pow)
        )
        
        total_opp_cnn_score = (
            ((opp_edges_mask & ~opp_corners_mask).bit_count() * self.my_edge_pow) +
            ((opp_corners_mask & ~opp_edges_mask).bit_count() * self.my_corner_pow) +
            ((opp_edges_mask & opp_corners_mask).bit_count() * self.my_both_pow)
        )

        # --- PHASE 4: SOFT-CAP NÉN PHÂN TẦNG VÀ ĐIỀU TỐC PHÒNG NGỰ ---
        my_soft = total_my_cnn_score / (total_my_cnn_score + self.c_smooth) if total_my_cnn_score > 0 else 0
        opp_soft = total_opp_cnn_score / (total_opp_cnn_score + self.c_smooth) if total_opp_cnn_score > 0 else 0
        
        alpha = self.alpha_balanced if total_my_cnn_score >= total_opp_cnn_score else self.alpha_defensive
        
        strategic_score = (self.max_strat * my_soft) - (self.max_strat * opp_soft * alpha)
        parity_score = (my_parity_threats * self.threat_score) - (opp_parity_threats * self.threat_score_opp)
        
        return int(strategic_score + parity_score + center_score)