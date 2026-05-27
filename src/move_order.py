
class MoveSorter:
    def __init__(self, board_width: int):
        """Bộ điều phối và sắp xếp nước đi thông minh sử dụng phân tầng Tuple"""
        self.w = board_width
        
        # Hệ thống bộ nhớ cho Killer và History Heuristics
        self.killer_moves = [[None, None] for _ in range(64)]  
        self.history_table = [0] * board_width  

    def clear_history(self):
        """
        🚀 TỐI ƯU: Lão hóa lịch sử (History Aging)
        Thay vì reset về 0 (làm mù AI ở rễ mới), ta chia đôi điểm lịch sử.
        """
        for i in range(self.w):
            self.history_table[i] //= 2

    def get_ordered_moves(self, board: ConnectXBoard, tt_move: int = None, current_depth: int = 1) -> list:
        """
        Trả về danh sách các cột hợp lệ đã được sắp xếp theo trật tự ưu tiên cố định:
        1. Kim chỉ nam tuyệt đối từ Bảng băm (TT Move) / Opening Book
        2. Các nước đi sát cục tiềm năng (Killer Moves)
        3. Chỉ số lịch sử hiệu quả (History Table)
        """
        valid_cols = board.get_valid_cols()
        if not valid_cols:
            return []

        k1 = self.killer_moves[current_depth][0]
        k2 = self.killer_moves[current_depth][1]

        def sort_key(col):
            return (
                col == tt_move,             # 1. Quyền ưu tiên tối thượng (TT Move / Book)
                col == k1,                  # 2. Ưu tiên thứ hai: Killer Move 1
                col == k2,                  # 3. Ưu tiên thứ ba: Killer Move 2
                self.history_table[col]     # 4. Ưu tiên cuối cùng: Điểm số từ History Table
            )

        valid_cols.sort(key=sort_key, reverse=True)
        return valid_cols