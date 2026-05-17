# evaluator.py
from board import ConnectXBoard

class BitboardCNNEvaluator:
    def __init__(self, config_weights):
        """Bộ lượng giá Bitwise CNN hình học - Bản nâng cấp sửa lỗi tràn bit cộng dồn"""
        self.weights = config_weights

    def evaluate(self, board: ConnectXBoard, current_player_id: int) -> int:
        my_id = current_player_id
        opp_id = 1 - my_id
        
        col_height = board.col_height
        
        # --- PHASE 1: GIẢ LẬP 1-STEP THREATS & FORK DETECTION ---
        my_threats = 0
        opp_threats = 0
        my_parity_threats = 0
        opp_parity_threats = 0
        
        valid_cols = board.get_valid_cols()
        
        for col in valid_cols:
            board.make_move(col, my_id)
            if board.check_win(my_id):
                my_threats += 1
                row = (board.heights[col] - 1) % col_height
                if (my_id == 0 and row % 2 == 0) or (my_id == 1 and row % 2 != 0):
                    my_parity_threats += 1
            board.undo_move(col, my_id)
            
            board.make_move(col, opp_id)
            if board.check_win(opp_id):
                opp_threats += 1
                row = (board.heights[col] - 1) % col_height
                if (opp_id == 0 and row % 2 == 0) or (opp_id == 1 and row % 2 != 0):
                    opp_parity_threats += 1
            board.undo_move(col, opp_id)

        if opp_threats >= 2:
            return -int(self.weights["FORK_SCORE"])
        if my_threats >= 2:
            return int(self.weights["FORK_SCORE"])

        # --- PHASE 2: THƯỞNG ĐIỂM CHIẾM TRỤC TRUNG TÂM (CENTER CONTROL) ---
        center_col = board.w // 2
        center_mask = ((1 << board.h) - 1) << (center_col * col_height)
        
        flank_mask = 0
        if center_col - 1 >= 0:
            flank_mask |= ((1 << board.h) - 1) << ((center_col - 1) * col_height)
        if center_col + 1 < board.w:
            flank_mask |= ((1 << board.h) - 1) << ((center_col + 1) * col_height)

        my_center_bits = bin(board.boards[my_id] & center_mask).count('1')
        opp_center_bits = bin(board.boards[opp_id] & center_mask).count('1')
        my_flank_bits = bin(board.boards[my_id] & flank_mask).count('1')
        opp_flank_bits = bin(board.boards[opp_id] & flank_mask).count('1')

        center_score = (my_center_bits - opp_center_bits) * 150 + (my_flank_bits - opp_flank_bits) * 50

        # --- PHASE 3: VECTOR HÓA CNN FEATURE MAP (BỎ WHILE LOOP, DÙNG BITWISE OR) ---
        empty_mask = (~(board.boards[0] | board.boards[1])) & board.valid_mask
        
        my_board = board.boards[my_id]
        opp_board = board.boards[opp_id]
        
        # ĐÃ SỬA: Thay toán tử cộng (+) bằng toán tử OR (|) để triệt tiêu lỗi rò rỉ bit
        my_edges_mask = 0
        for s in board.shifts[:2]:  # 1 và col_height (Ngang/Dọc)
            my_edges_mask |= ((my_board << s) & board.valid_mask) | ((my_board >> s) & board.valid_mask)
        my_corners_mask = 0
        for s in board.shifts[2:]:  # Đường chéo thuận/nghịch
            my_corners_mask |= ((my_board << s) & board.valid_mask) | ((my_board >> s) & board.valid_mask)
            
        opp_edges_mask = 0
        for s in board.shifts[:2]:
            opp_edges_mask |= ((opp_board << s) & board.valid_mask) | ((opp_board >> s) & board.valid_mask)
        opp_corners_mask = 0
        for s in board.shifts[2:]:
            opp_corners_mask |= ((opp_board << s) & board.valid_mask) | ((opp_board >> s) & board.valid_mask)

        # Lọc không gian ô trống
        my_edges_mask &= empty_mask
        my_corners_mask &= empty_mask
        opp_edges_mask &= empty_mask
        opp_corners_mask &= empty_mask

        # Phân tách 3 nhóm ô độc lập để tính toán mật độ cụm quân lũy thừa
        my_both = my_edges_mask & my_corners_mask
        opp_both = opp_edges_mask & opp_corners_mask
        
        my_edge_only = my_edges_mask & ~my_corners_mask
        opp_edge_only = opp_edges_mask & ~opp_corners_mask
        
        my_corner_only = my_corners_mask & ~my_edges_mask
        opp_corner_only = opp_corners_mask & ~opp_edges_mask

        total_my_cnn_score = (
            (bin(my_edge_only).count('1') * (self.weights["K_EDGE"] ** self.weights["CNN_POWER"])) +
            (bin(my_corner_only).count('1') * (self.weights["K_CORNER"] ** self.weights["CNN_POWER"])) +
            (bin(my_both).count('1') * ((self.weights["K_EDGE"] + self.weights["K_CORNER"]) ** self.weights["CNN_POWER"]))
        )
        
        total_opp_cnn_score = (
            (bin(opp_edge_only).count('1') * (self.weights["K_EDGE"] ** self.weights["CNN_POWER"])) +
            (bin(opp_corner_only).count('1') * (self.weights["K_CORNER"] ** self.weights["CNN_POWER"])) +
            (bin(opp_both).count('1') * ((self.weights["K_EDGE"] + self.weights["K_CORNER"]) ** self.weights["CNN_POWER"]))
        )

        # --- PHASE 4: SOFT-CAP NÉN PHÂN TẦNG VÀ ĐIỀU TỐC PHÒNG NGỰ ---
        C = self.weights["C_SMOOTH"]
        MAX_STRAT = self.weights["MAX_STRATEGIC"]
        
        my_soft = total_my_cnn_score / (total_my_cnn_score + C) if total_my_cnn_score > 0 else 0
        opp_soft = total_opp_cnn_score / (total_opp_cnn_score + C) if total_opp_cnn_score > 0 else 0
        
        alpha = self.weights["ALPHA_BALANCED"] if total_my_cnn_score >= total_opp_cnn_score else self.weights["ALPHA_DEFENSIVE"]
        
        strategic_score = (MAX_STRAT * my_soft) - (MAX_STRAT * opp_soft * alpha)
        parity_score = (my_parity_threats * self.weights["THREAT_SCORE"]) - (opp_parity_threats * self.weights["THREAT_SCORE"] * 1.2)
        
        return int(strategic_score + parity_score + center_score)