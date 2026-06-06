# move_order.py

class MoveSorter:
    def __init__(self, board_width: int):
        """Bộ điều phối và sắp xếp nước đi thông minh sử dụng phân tầng Tuple"""
        self.w = board_width
        
        # Hệ thống bộ nhớ cho Killer và History Heuristics
        self.killer_moves = [[None, None] for _ in range(64)]  # Hỗ trợ tối đa 64 tầng ply đệ quy
        self.history_table = [0] * board_width  # Mảng 1D lưu điểm lịch sử cho ConnectX (theo cột)

    def clear_history(self):
        """Reset bảng lịch sử trước mỗi lượt đi mới của Iterative Deepening"""
        self.history_table = [0] * self.w

    def get_ordered_moves(self, board, valid_cols: list, tt_move: int, ply: int, current_player_id: int, last_move: int = None, depth: int = 0) -> list:
        """Sắp xếp nước đi bằng cơ chế Tuple-Priority, không dùng magic numbers, không dùng remove/insert"""
        if len(valid_cols) <= 1:
            return valid_cols

        my_id = current_player_id
        opp_id = 1 - my_id

        # Lấy cặp nước đi Killer ở tầng ply hiện tại
        k1, k2 = self.killer_moves[ply][0], self.killer_moves[ply][1]
        
        # Xác định tâm mỏ neo để tính điểm khoảng cách hình học
        target_center = last_move if last_move is not None else (self.w // 2)

        # TỐI ƯU CHI PHÍ: Chỉ tốn tài nguyên giả lập 1-step tactical (make/undo move) khi độ sâu còn lại đủ lớn.
        # Ở các node sát lá hoặc trong Quiescence Search (depth thấp), bỏ qua check này để giữ max tốc độ NPS.
        check_tactical = (depth >= 3)

        def sort_key(col):
            is_my_win = False
            is_opp_win = False

            if check_tactical:
                # Giả lập nhanh trạng thái 1-step chiến thuật để lấy thông tin Tầng cao
                board.make_move(col, my_id)
                is_my_win = board.check_win(my_id)
                board.undo_move(col, my_id)
                
                if not is_my_win:
                    board.make_move(col, opp_id)
                    is_opp_win = board.check_win(opp_id)
                    board.undo_move(col, opp_id)

            # Trả về Tuple phân tầng ưu tiên (True đứng trước False khi reverse=True)
            return (
                col == tt_move,             # 1. Ưu tiên số 1: Nước đi chiến lược từ bảng băm TT
                is_my_win,                  # 2. Ưu tiên số 2: Nước đi thắng ngay (Tầng 0)
                is_opp_win,                 # 3. Ưu tiên số 3: Nước đi cứu mạng phải chặn ngay (Tầng 0)
                col == k1,                  # 4. Ưu tiên số 4: Killer Move slot 1
                col == k2,                  # 5. Ưu tiên số 5: Killer Move slot 2
                self.history_table[col],    # 6. Ưu tiên số 6: Điểm History (Tích lũy từ quá khứ)
                (self.w - abs(col - target_center)) # 7. Ưu tiên số 7: Điểm hình học mỏ neo tâm bàn cờ
            )

        return sorted(valid_cols, key=sort_key, reverse=True)