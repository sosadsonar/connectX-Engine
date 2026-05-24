# evaluator.py
from board import ConnectXBoard

class BitboardCNNEvaluator:
    def __init__(self, config_weights):
        """
        Bộ lượng giá tối ưu hóa cấu trúc hình học Connect 4 (7x6):
        - Bảo toàn tính nghiêm ngặt tuyệt đối của THREAT_SCORE và FORK_SCORE.
        - Tích hợp Radar Parity động quét theo từng hàng tranh chấp cụ thể.
        - Tích hợp cơ chế Decay (phai nhạt) điểm kiểm soát trung tâm ở đầu game.
        """
        self.weights = config_weights
        self.is_initialized = False
        
        # Cache các hằng số lũy thừa hình học CNN
        self.my_edge_pow = self.weights["K_EDGE"] ** self.weights["CNN_POWER"]
        self.my_corner_pow = self.weights["K_CORNER"] ** self.weights["CNN_POWER"]
        self.my_both_pow = (self.weights["K_EDGE"] + self.weights["K_CORNER"]) ** self.weights["CNN_POWER"]
        
        # Cache các trọng số cốt lõi từ bộ gen
        self.fork_score = int(self.weights["FORK_SCORE"])
        self.threat_score = self.weights["THREAT_SCORE"]
        self.threat_score_opp = self.weights["THREAT_SCORE"] * 1.2
        self.c_smooth = self.weights["C_SMOOTH"]
        self.max_strat = self.weights["MAX_STRATEGIC"]
        self.alpha_balanced = self.weights["ALPHA_BALANCED"]
        self.alpha_defensive = self.weights["ALPHA_DEFENSIVE"]
        self.win_base = self.weights["WIN_BASE"]

        # Cache các trọng số Parity và Trung tâm mở rộng mới
        self.center_col_weight = self.weights.get("CENTER_COL_WEIGHT", 450.0)
        self.parity_row_bonus = self.weights.get("PARITY_ROW_BONUS", 150.0)
        self.parity_row_penalty = self.weights.get("PARITY_ROW_PENALTY", 180.0)

    def _lazy_init(self, board: ConnectXBoard):
        """Khởi tạo một lần duy nhất các mặt nạ dịch bit cố định"""
        self.col_height = board.col_height
        center_col = board.w // 2
        
        # Mặt nạ kiểm soát cột trung tâm (Cột 3)
        self.center_mask = ((1 << board.h) - 1) << (center_col * self.col_height)
        
        # Mặt nạ các cột cánh liền kề trung tâm (Cột 2, 4)
        self.flank_mask = 0
        if center_col - 1 >= 0:
            self.flank_mask |= ((1 << board.h) - 1) << ((center_col - 1) * self.col_height)
        if center_col + 1 < board.w:
            self.flank_mask |= ((1 << board.h) - 1) << ((center_col + 1) * self.col_height)
            
        # Lưới quét chẵn lẻ tĩnh phục vụ đếm Threat đơn Playable ở Phase 4
        self.bottom_mask = 0
        self.even_rows_mask = 0
        self.odd_rows_mask = 0
        for c in range(board.w):
            self.bottom_mask |= (1 << (c * self.col_height))
            for r in range(0, board.h, 2):
                self.even_rows_mask |= (1 << (c * self.col_height + r))
            for r in range(1, board.h, 2):
                self.odd_rows_mask |= (1 << (c * self.col_height + r))
                
        self.is_initialized = True

    def _get_winning_spots(self, P: int, shifts: list, valid_mask: int, x: int) -> int:
        """Ma thuật Bitwise quét điểm suýt thắng (Check tốt mọi kiểu cách ô, trống 1 hổng giữa)"""
        threats = 0
        if x == 4:
            for s in shifts:
                p_right2 = (P >> s) & (P >> 2*s)
                p_left2 = (P << s) & (P << 2*s)
                threats |= p_right2 & ((P >> 3*s) | (P << s))
                threats |= p_left2 & ((P << 3*s) | (P >> s))
            return threats & valid_mask
            
        for s in shifts:
            for hole_pos in range(x):
                mask = valid_mask
                for i in range(x):
                    if i == hole_pos: continue
                    dist = i - hole_pos
                    if dist > 0: mask &= (P >> (dist * s))
                    else: mask &= (P << (-dist * s))
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
        
        # --- PHASE 1: PHÁT HIỆN SÁT CỤC ĐƠN ĐỂ KIỂM SOÁT FORK ---
        playable_mask = (occupied + self.bottom_mask) & board.valid_mask
        
        my_all_winning_spots = self._get_winning_spots(my_board, board.shifts, board.valid_mask, board.x)
        opp_all_winning_spots = self._get_winning_spots(opp_board, board.shifts, board.valid_mask, board.x)
        
        my_immediate_threats = my_all_winning_spots & playable_mask
        opp_immediate_threats = opp_all_winning_spots & playable_mask
        
        my_threats_count = my_immediate_threats.bit_count()
        opp_threats_count = opp_immediate_threats.bit_count()
        
        # Ngắt mạch khẩn cấp (Short-Circuit): Nếu dính đòn Fork, trả điểm cấu trúc cấp cao và thoát sớm
        if opp_threats_count >= 2: return -self.fork_score
        if my_threats_count >= 2: return self.fork_score

        # --- PHASE 2: ĐỊNH LƯỢNG VỊ TRÍ PARITY ĐỘNG THEO TỪNG HÀNG TRANH CHẤP ---
        parity_positional_score = 0
        for r in range(board.h):
            row_mask = 0
            for c in range(board.w):
                row_mask |= (1 << (c * self.col_height + r))
            
            my_row_count = (my_board & row_mask).bit_count()
            opp_row_count = (opp_board & row_mask).bit_count()
            total_row_pieces = my_row_count + opp_row_count
            
            # Hệ số động: Hàng càng dày quân -> Tranh chấp càng cao -> Nhân mạnh điểm
            dynamic_multiplier = 1.0 + (total_row_pieces * 0.5)
            
            if r % 2 == 1:  # Hàng Lẻ (Odd Row)
                if my_id == 0:  # Bot đi trước (X) -> Muốn làm chủ hàng lẻ
                    parity_positional_score += int(my_row_count * self.parity_row_bonus * dynamic_multiplier)
                else:          # Bot đi sau (O) -> Phạt nếu để đối thủ chiếm nhiều hàng lẻ
                    parity_positional_score -= int(opp_row_count * self.parity_row_penalty * dynamic_multiplier)
            else:          # Hàng Chẵn (Even Row)
                if my_id == 1:  # Bot đi sau (O) -> Muốn làm chủ hàng chẵn
                    parity_positional_score += int(my_row_count * self.parity_row_bonus * dynamic_multiplier)
                else:          # Bot đi trước (X) -> Phạt nếu để đối thủ chiếm nhiều hàng chẵn
                    parity_positional_score -= int(opp_row_count * self.parity_row_penalty * dynamic_multiplier)

        # --- PHASE 3: THƯỞNG ĐIỂM TRUNG TÂM TUYỆT ĐỐI (2 NƯỚC ĐẦU) & DECAY ĐỘNG (GIAI ĐOẠN SAU) ---
        my_center_bits = (my_board & self.center_mask).bit_count()
        opp_center_bits = (opp_board & self.center_mask).bit_count()
        my_flank_bits = (my_board & self.flank_mask).bit_count()
        opp_flank_bits = (opp_board & self.flank_mask).bit_count()

        total_pieces = occupied.bit_count()
        
        if total_pieces <= 5:
            opening_multiplier = 2.0
        else:
            opening_multiplier = max(0.15, 1.0 - (total_pieces - 5) * 0.05)


        raw_center_score = (my_center_bits - opp_center_bits) * self.center_col_weight + (my_flank_bits - opp_flank_bits) * 50
        center_score = int(raw_center_score * opening_multiplier)

        # --- PHASE 4: FEATURE MAP CNN & ĐIỀU TỐC ĐE DỌA ĐƠN THREAT_SCORE ---
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

        my_edges_mask &= empty_mask; my_corners_mask &= empty_mask
        opp_edges_mask &= empty_mask; opp_corners_mask &= empty_mask

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

        my_soft = total_my_cnn_score / (total_my_cnn_score + self.c_smooth) if total_my_cnn_score > 0 else 0
        opp_soft = total_opp_cnn_score / (total_opp_cnn_score + self.c_smooth) if total_opp_cnn_score > 0 else 0
        
        alpha = self.alpha_balanced if total_my_cnn_score >= total_opp_cnn_score else self.alpha_defensive
        strategic_score = (self.max_strat * my_soft) - (self.max_strat * opp_soft * alpha)
        
        # Lượng giá sát cục đơn THREAT_SCORE dựa trên tính chẵn lẻ của ô trống Playable
        my_parity_mask = self.even_rows_mask if my_id == 0 else self.odd_rows_mask
        opp_parity_mask = self.even_rows_mask if opp_id == 0 else self.odd_rows_mask
        
        my_parity_threats = (my_immediate_threats & my_parity_mask).bit_count()
        opp_parity_threats = (opp_immediate_threats & opp_parity_mask).bit_count()
        
        parity_threat_score = (my_parity_threats * self.threat_score) - (opp_parity_threats * self.threat_score_opp)
        
        return int(strategic_score + parity_threat_score + center_score + parity_positional_score)