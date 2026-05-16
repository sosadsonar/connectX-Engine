class MinimaxAI:
    def __init__(self, max_depth, evaluator):
        self.max_depth = max_depth
        self.evaluator = evaluator

    def choose_best_move(self, board_obj, player_id):
        """Hàm bề nổi để Bot chọn ra cột tốt nhất mang lại điểm số cao nhất"""
        valid_cols = board_obj.get_valid_cols()
        best_score = -float('inf')
        best_col = None
        
        for col in valid_cols:
            board_obj.make_move(col, player_id)
            score = self._minimax(board_obj, self.max_depth - 1, player_id, is_maximizing=False)
            board_obj.undo_move(col, player_id)
            
            if score > best_score:
                best_score = score
                best_col = col
        return best_col

    def _minimax(self, board_obj, depth, player_id, is_maximizing):
        opp_id = 1 - player_id
        valid_cols = board_obj.get_valid_cols()

        # Kiểm tra các trạng thái kết thúc (Terminal Nodes)
        if board_obj.check_win(player_id):
            return 10000000
        if board_obj.check_win(opp_id):
            return -10000000
        if len(valid_cols) == 0:
            return 0
        if depth == 0:
            return self.evaluator.evaluate(board_obj, player_id)

        if is_maximizing:
            value = -float('inf')
            for col in valid_cols:
                board_obj.make_move(col, player_id)
                score = self._minimax(board_obj, depth - 1, player_id, False)
                board_obj.undo_move(col, player_id)
                value = max(value, score)
            return value
        else:
            value = float('inf')
            for col in valid_cols:
                board_obj.make_move(col, opp_id)
                score = self._minimax(board_obj, depth - 1, player_id, True)
                board_obj.undo_move(col, opp_id)
                value = min(value, score)
            return value
