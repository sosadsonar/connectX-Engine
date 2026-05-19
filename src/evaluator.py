# evaluator.py
from board import ConnectXBoard

class BitboardCNNEvaluator:
    def __init__(self, config_weights):
        """Bộ lượng giá Bitwise CNN hình học - Bản tối ưu hóa Zero-Simulation tĩnh 100%"""
        self.weights = config_weights
        
        self.is_initialized = False
        
        # Khởi tạo sẵn các hằng số lũy thừa
        self.my_edge_pow = self.weights["K_EDGE"] ** self.weights["CNN_POWER"]
        self.my_corner_pow = self.weights["K_CORNER"] ** self.weights["CNN_POWER"]
        self.my_both_pow = (self.weights["K_EDGE"] + self.weights["K_CORNER"]) ** self.weights["CNN_POWER"]
        
        # Cache các trọng số
        self.fork_score = int(self.weights["FORK_SCORE"])
        self.threat_score = self.weights["THREAT_SCORE"]
        self.threat_score_opp = self.weights["THREAT_SCORE"] * 1.2
        self.c_smooth = self.weights["C_SMOOTH"]
        self.max_strat = self.weights["MAX_STRATEGIC"]
        self.alpha_balanced = self.weights["ALPHA_BALANCED"]
        self.alpha_defensive = self.weights["ALPHA_DEFENSIVE"]
        self.win_base = self.weights["WIN_BASE"]

    def _lazy_init(self, board: ConnectXBoard):
        """Khởi tạo một lần duy nhất các mặt nạ dịch bit cố định"""
        self.col_height = board.col_height
        center_col = board.w // 2
        
        self.center_mask = ((1 << board.h) - 1) << (center_col * self.col_height)
        
        self.flank_mask = 0
        if center_col - 1 >= 0:
            self.flank_mask |= ((1 << board.h) - 1) << ((center_col - 1) * self.col_height)
        if center_col + 1 < board.w:
            self.flank_mask |= ((1 << board.h) - 1) << ((center_col + 1) * self.col_height)
            
        # 🚀 CẤU TRÚC LƯỚI TOÁN HỌC DÀNH CHO THREAT MASK
        self.bottom_mask = 0
        self.even_rows_mask = 0
        self.odd_rows_mask = 0
        for c in range(board.w):
            # Lấy toàn bộ hàng đáy (Row 0)
            self.bottom_mask |= (1 << (c * self.col_height))
            # Quét mặt nạ dòng chẵn (Row 0, 2, 4...)
            for r in range(0, board.h, 2):
                self.even_rows_mask |= (1 << (c * self.col_height + r))
            # Quét mặt nạ dòng lẻ (Row 1, 3, 5...)
            for r in range(1, board.h, 2):
                self.odd_rows_mask |= (1 << (c * self.col_height + r))
                
        self.is_initialized = True

    def _get_winning_spots(self, P: int, shifts: list, valid_mask: int, x: int) -> int:
        """
        🚀 MA THUẬT BITWISE VẠN NĂNG: Hỗ trợ mọi luật Connect X.
        """
        threats = 0
        
        # ⚡ TỐI ƯU CỨNG CHO CONNECT 4 (Chạy tốc độ ánh sáng O(1))
        if x == 4:
            for s in shifts:
                p_right2 = (P >> s) & (P >> 2*s)
                p_left2 = (P << s) & (P << 2*s)
                threats |= p_right2 & ((P >> 3*s) | (P << s))
                threats |= p_left2 & ((P << 3*s) | (P >> s))
            return threats & valid_mask
            
        # 🛡️ CƠ CHẾ DỰ PHÒNG ĐỘNG (Dành cho Connect 3, Connect 5, Connect 6...)
        # Quét trượt (Sliding window) trên hệ quy chiếu Bitwise
        for s in shifts:
            # Lặp qua từng vị trí có thể đặt "lỗ hổng" trong chuỗi X ô
            for hole_pos in range(x):
                mask = valid_mask
                # Yêu cầu (X - 1) ô còn lại đều phải có quân của phe ta
                for i in range(x):
                    if i == hole_pos:
                        continue
                    dist = i - hole_pos
                    if dist > 0:
                        mask &= (P >> (dist * s))
                    else:
                        mask &= (P << (-dist * s))
                threats |= mask
                
        return threats & valid_mask

    def evaluate(self, board: ConnectXBoard, current_player_id: int) -> int:
        if not self.is_initialized:
            self._lazy_init(board)

        my_id = current_player_id
        opp_id = 1 - my_id
        
        my_board = board.boards[my_id]
        opp_board = board.boards[opp_id]
        occupied = my_board | opp_board
        
        # --- PHASE 1: CHỤP X-QUANG ĐE DỌA 1-STEP (ZERO SIMULATION) ---
        
        # 1. Playable Mask: Xác định chính xác các ô "chân lơ lửng" có thể thả quân vào ngay lúc này
        # (Cộng nguyên mảng bit với hàng đáy sẽ đẩy bit lên đúng 1 nấc lọt vào ô trống tiếp theo)
        playable_mask = (occupied + self.bottom_mask) & board.valid_mask
        
        # 2. Winning Spots: Quét toàn bộ điểm mù dọa sát cục của hai phe
        # (Nằm ở mục 2 của Phase 1 trong hàm evaluate)
        my_all_winning_spots = self._get_winning_spots(my_board, board.shifts, board.valid_mask, board.x)
        opp_all_winning_spots = self._get_winning_spots(opp_board, board.shifts, board.valid_mask, board.x)
        
        # 3. Kéo mảng giao nhau: Những điểm sát cục nằm TRÚNG vào ô có thể đi ngay lập tức
        my_immediate_threats = my_all_winning_spots & playable_mask
        opp_immediate_threats = opp_all_winning_spots & playable_mask
        
        my_threats_count = my_immediate_threats.bit_count()
        opp_threats_count = opp_immediate_threats.bit_count()
        
        # Phát hiện Fork vĩ mô siêu tốc
        if opp_threats_count >= 2: return -self.fork_score
        if my_threats_count >= 2: return self.fork_score

        # 4. Trích xuất Tính Chẵn Lẻ (Parity) trực tiếp bằng AND logic
        my_parity_mask = self.even_rows_mask if my_id == 0 else self.odd_rows_mask
        opp_parity_mask = self.even_rows_mask if opp_id == 0 else self.odd_rows_mask
        
        my_parity_threats = (my_immediate_threats & my_parity_mask).bit_count()
        opp_parity_threats = (opp_immediate_threats & opp_parity_mask).bit_count()

        # --- PHASE 2: THƯỞNG ĐIỂM CHIẾM TRỤC TRUNG TÂM ---
        my_center_bits = (my_board & self.center_mask).bit_count()
        opp_center_bits = (opp_board & self.center_mask).bit_count()
        my_flank_bits = (my_board & self.flank_mask).bit_count()
        opp_flank_bits = (opp_board & self.flank_mask).bit_count()

        center_score = (my_center_bits - opp_center_bits) * 150 + (my_flank_bits - opp_flank_bits) * 50

        # --- PHASE 3: VECTOR HÓA CNN FEATURE MAP ---
        empty_mask = (~occupied) & board.valid_mask
        
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