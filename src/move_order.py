# move_order.py

class MoveSorter:
    def __init__(self, board_width: int):
        """Bộ điều phối và sắp xếp nước đi thông minh sử dụng phân tầng Tuple"""
        self.w = board_width
        
        # Hệ thống bộ nhớ cho Killer và History Heuristics
        self.killer_moves = [[None, None] for _ in range(64)]  
        self.history_table = [0] * board_width  

    def clear_history(self):
        """
        🚀 TỐI ƯU 2: Lão hóa lịch sử (History Aging)
        Thay vì reset về 0 (làm mù AI ở rễ mới), ta chia đôi điểm lịch sử.
        Giúp AI nhớ được các khu vực điểm nóng của lượt trước nhưng không bị lạm phát điểm.
        """
        for i in range(self.w):
            self.history_table[i] //= 2

    def get_ordered_moves(self, board, valid_cols: list, tt_move: int, ply: int, current_player_id: int, last_move: int = None, depth: int = 0) -> list:
        """Sắp xếp nước đi siêu tốc: Loại bỏ hoàn toàn overhead gọi hàm giả lập make/undo"""
        if len(valid_cols) <= 1:
            return valid_cols

        # Lấy cặp nước đi Killer ở tầng ply hiện tại
        k1, k2 = self.killer_moves[ply][0], self.killer_moves[ply][1]
        
        # Xác định tâm mỏ neo để tính điểm khoảng cách hình học
        target_center = last_move if last_move is not None else (self.w // 2)

        # 🚀 TỐI ƯU 1: Sort Key thuần túy là truy xuất thuộc tính phẳng (O(1))
        # Không có bất kỳ vòng lặp hay lời gọi hàm đắt đỏ nào ở đây
        def sort_key(col):
            return (
                col == tt_move,             # 1. Bảng băm (Transposition Table) là kim chỉ nam tuyệt đối
                col == k1,                  # 2. Killer Move 1 (Cắt tỉa nhánh ngang cực mạnh)
                col == k2,                  # 3. Killer Move 2
                self.history_table[col],    # 4. History Score (Điểm uy tín từ các nhánh đã duyệt)
                -abs(col - target_center)   # 5. Hình học: Càng gần rốn trung tâm càng tốt (Dùng dấu trừ để reverse mượt)
            )

        return sorted(valid_cols, key=sort_key, reverse=True)