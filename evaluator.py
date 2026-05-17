# evaluator.py
from board import ConnectXBoard

class BitboardCNNEvaluator:
    def __init__(self, config_weights):
        self.weights = config_weights

    def evaluate(self, board: ConnectXBoard, current_player_id: int) -> int:
        my_id = current_player_id
        opp_id = 1 - my_id
        
        X = board.x
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
                last_bit = board.heights[col] - 1
                row = last_bit % col_height
                if (my_id == 0 and row % 2 == 0) or (my_id == 1 and row % 2 != 0):
                    my_parity_threats += 1
            board.undo_move(col, my_id)
            
            board.make_move(col, opp_id)
            if board.check_win(opp_id):
                opp_threats += 1
                last_bit = board.heights[col] - 1
                row = last_bit % col_height
                if (opp_id == 0 and row % 2 == 0) or (opp_id == 1 and row % 2 != 0):
                    opp_parity_threats += 1
            board.undo_move(col, opp_id)

        if opp_threats >= 2:
            return -int(self.weights["FORK_SCORE"])
        if my_threats >= 2:
            return int(self.weights["FORK_SCORE"])

        # --- PHASE 2: TRÍCH XUẤT ĐẶC TRƯNG MẬT ĐỘ (CNN BITWISE KERNEL) ---
        empty_mask = (~(board.boards[0] | board.boards[1])) & board.valid_mask
        
        SHIFT_UP_DOWN = [1, board.col_height]
        SHIFT_DIAGONALS = [board.col_height + 1, board.col_height - 1]
        
        my_board = board.boards[my_id]
        opp_board = board.boards[opp_id]
        
        my_edges_map = 0
        for s in SHIFT_UP_DOWN:
            my_edges_map += ((my_board << s) & board.valid_mask) + ((my_board >> s) & board.valid_mask)
            
        my_corners_map = 0
        for s in SHIFT_DIAGONALS:
            my_corners_map += ((my_board << s) & board.valid_mask) + ((my_board >> s) & board.valid_mask)
            
        opp_edges_map = 0
        for s in SHIFT_UP_DOWN:
            opp_edges_map += ((opp_board << s) & board.valid_mask) + ((opp_board >> s) & board.valid_mask)
            
        opp_corners_map = 0
        for s in SHIFT_DIAGONALS:
            opp_corners_map += ((opp_board << s) & board.valid_mask) + ((opp_board >> s) & board.valid_mask)

        total_my_cnn_score = 0.0
        total_opp_cnn_score = 0.0
        
        temp_empty = empty_mask
        while temp_empty > 0:
            idx = (temp_empty & -temp_empty).bit_length() - 1
            temp_empty &= temp_empty - 1
            
            bit_flag = (1 << idx)
            
            my_edges = 1 if (my_edges_map & bit_flag) else 0
            my_corners = 1 if (my_corners_map & bit_flag) else 0
            my_density = (my_edges * self.weights["K_EDGE"]) + (my_corners * self.weights["K_CORNER"])
            if my_density > 0:
                total_my_cnn_score += my_density ** self.weights["CNN_POWER"]
                
            opp_edges = 1 if (opp_edges_map & bit_flag) else 0
            opp_corners = 1 if (opp_corners_map & bit_flag) else 0
            opp_density = (opp_edges * self.weights["K_EDGE"]) + (opp_corners * self.weights["K_CORNER"])
            if opp_density > 0:
                total_opp_cnn_score += opp_density ** self.weights["CNN_POWER"]

        # --- PHASE 3: SOFT-CAP NÉN PHÂN TẦNG VÀ TÍNH ĐIỂM TỔNG HỢP ---
        C = self.weights["C_SMOOTH"]
        MAX_STRAT = self.weights["MAX_STRATEGIC"]
        
        my_soft = total_my_cnn_score / (total_my_cnn_score + C) if total_my_cnn_score > 0 else 0
        opp_soft = total_opp_cnn_score / (total_opp_cnn_score + C) if total_opp_cnn_score > 0 else 0
        
        alpha = self.weights["ALPHA_BALANCED"] if total_my_cnn_score >= total_opp_cnn_score else self.weights["ALPHA_DEFENSIVE"]
        
        strategic_score = (MAX_STRAT * my_soft) - (MAX_STRAT * opp_soft * alpha)
        parity_score = (my_parity_threats * self.weights["THREAT_SCORE"]) - (opp_parity_threats * self.weights["THREAT_SCORE"] * 1.2)
        
        return int(strategic_score + parity_score)