import itertools

class BoardEvaluator:
    def __init__(self, win_condition=4):
        self.window_size = win_condition
        self.weights = {
            "my_3": 100,
            "my_2": 5,
            "opp_3": 10000,
            "opp_2": 10
        }

    def _count_pattern(self, P, E, shift, num_pieces):
        """
        HÀM THẦN KỲ: Đếm số cửa sổ chứa đúng 'num_pieces' quân cờ và còn lại là ô trống.
        Tự động sinh cấu trúc dịch bit mà không cần dùng vòng lặp quét ô cờ bừa bãi.
        """
        total_count = 0
        # Tự động sinh vị trí các ô cờ trong một cửa sổ (Ví dụ: 3 quân cờ xếp ở vị trí nào)
        for piece_indices in itertools.combinations(range(self.window_size), num_pieces):
            combined = None
            for i in range(self.window_size):
                # Nếu i nằm trong nhóm đặt quân thì lấy bảng P, ngược lại lấy bảng trống E
                current_board = P if i in piece_indices else E
                shifted = current_board >> (i * shift)
                
                if combined is None:
                    combined = shifted
                else:
                    combined &= shifted
            total_count += combined.bit_count()
        return total_count

    def evaluate(self, board_obj, player_id):
        """Tính tổng điểm của bàn cờ hiện tại dưới góc nhìn của player_id"""
        opp_id = 1 - player_id
        P = board_obj.boards[player_id]
        O = board_obj.boards[opp_id]
        E = ~(P | O) & board_obj.valid_mask  # Toàn bộ ô trống hợp lệ
        
        score = 0
        for shift in board_obj.shifts:
            # Đếm chuỗi của mình
            my_3 = self._count_pattern(P, E, shift, num_pieces=3)
            my_2 = self._count_pattern(P, E, shift, num_pieces=2)
            
            # Đếm chuỗi của đối thủ
            opp_3 = self._count_pattern(O, E, shift, num_pieces=3)
            opp_2 = self._count_pattern(O, E, shift, num_pieces=2)
            
            # Cộng trừ điểm theo cấu hình trọng số
            score += (my_3 * self.weights["my_3"]) + (my_2 * self.weights["my_2"])
            score -= (opp_3 * self.weights["opp_3"]) + (opp_2 * self.weights["opp_2"])
            
        return score
