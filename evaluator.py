"""
evaluator.py

Module chấm điểm trạng thái hiện tại của ConnectXBoard.

Kiến trúc hiện tại:
- board.py: quản lý bitboard, make_move/undo_move, valid columns, check_win
- ai.py: minimax, gọi evaluator.evaluate(board_obj, player_id)
- evaluator.py: toàn bộ heuristic đánh giá board

Player id trong project hiện tại là 0 và 1.
Ô trống được biểu diễn gián tiếp qua valid_mask của board.
"""

import itertools


class BoardEvaluator:
    """
    Chấm điểm bàn cờ hiện tại dưới góc nhìn của player_id.

    Score > 0: tốt cho player_id
    Score < 0: tốt cho đối thủ
    Score càng lớn: minimax càng muốn chọn trạng thái đó
    """

    def __init__(self, win_condition=4):
        self.window_size = win_condition

        # Terminal score nên lớn hơn mọi điểm heuristic cộng lại.
        # ai.py hiện đang dùng 10_000_000 cho terminal node,
        # evaluator dùng thấp hơn một chút để không lấn logic terminal của minimax.
        self.WIN_SCORE = 9_000_000
        self.LOSE_SCORE = -9_000_000

        # Bộ trọng số mặc định cho Connect 4.
        # Có thể fine-tune bằng cách chỉnh các giá trị này.
        self.weights = {
            # Threat đánh thắng ngay ở nước tiếp theo.
            "my_immediate_win": 120_000,
            "opp_immediate_win": 180_000,

            # Window chỉ gồm quân mình + ô trống.
            # Ví dụ x=4: 3 quân + 1 trống, 2 quân + 2 trống, 1 quân + 3 trống.
            "my_3": 8_000,
            "my_2": 300,
            "my_1": 8,

            # Window chỉ gồm quân đối thủ + ô trống.
            # Phạt đối thủ thường mạnh hơn thưởng mình một chút để bot biết chặn.
            "opp_3": 12_000,
            "opp_2": 450,
            "opp_1": 10,

            # Ưu tiên kiểm soát trung tâm.
            "center": 12,
        }

    def _get_playable_mask(self, board_obj):
        """
        Trả về bitmask các ô có thể đặt quân NGAY ở lượt hiện tại.

        Với bitboard dạng cột, board_obj.heights[c] luôn trỏ tới ô trống thấp nhất
        của cột c. Nếu cột chưa đầy, bit đó là một nước đi hợp lệ.
        """
        playable = 0
        for c in board_obj.get_valid_cols():
            playable |= 1 << board_obj.heights[c]
        return playable

    def _count_pattern(self, piece_board, empty_board, shift, num_pieces):
        """
        Đếm số window có đúng num_pieces quân của piece_board,
        các ô còn lại thuộc empty_board.

        Ví dụ Connect 4, num_pieces=3:
            [P, P, P, E]
            [P, P, E, P]
            [P, E, P, P]
            [E, P, P, P]

        Hàm này dùng shift bitboard nên không cần quét từng ô bằng mảng 2D.
        """
        if num_pieces < 0 or num_pieces > self.window_size:
            return 0

        total_count = 0
        positions = range(self.window_size)

        for piece_indices in itertools.combinations(positions, num_pieces):
            piece_indices = set(piece_indices)
            combined = None

            for i in positions:
                source = piece_board if i in piece_indices else empty_board
                shifted = source >> (i * shift)
                combined = shifted if combined is None else (combined & shifted)

            total_count += combined.bit_count()

        return total_count

    def evaluate_window(self, own_count, opp_count, empty_count, playable_empty_count=0):
        """
        Chấm điểm logic cho MỘT loại window sau khi đã biết số quân.

        Vì evaluator đang chạy trên bitboard, ta không truyền list kiểu [1,1,0,0]
        vào đây nữa. Thay vào đó truyền số lượng:
        - own_count: số quân của mình trong window
        - opp_count: số quân đối thủ trong window
        - empty_count: số ô trống trong window
        - playable_empty_count: số ô trống có thể đánh ngay trong window

        Window bị chặn bởi cả hai bên thì không có giá trị chiến thuật.
        """
        if own_count > 0 and opp_count > 0:
            return 0

        if own_count == self.window_size:
            return self.WIN_SCORE
        if opp_count == self.window_size:
            return self.LOSE_SCORE

        score = 0

        # Mình có x-1 quân và ô còn lại có thể đánh ngay: nước thắng trực tiếp.
        if own_count == self.window_size - 1 and empty_count == 1:
            score += self.weights["my_3"]
            if playable_empty_count == 1:
                score += self.weights["my_immediate_win"]

        # Đối thủ có x-1 quân và ô còn lại có thể đánh ngay: phải chặn.
        elif opp_count == self.window_size - 1 and empty_count == 1:
            score -= self.weights["opp_3"]
            if playable_empty_count == 1:
                score -= self.weights["opp_immediate_win"]

        # Các thế phát triển chưa thắng ngay.
        elif own_count == self.window_size - 2 and empty_count == 2:
            score += self.weights["my_2"]
        elif opp_count == self.window_size - 2 and empty_count == 2:
            score -= self.weights["opp_2"]
        elif own_count == self.window_size - 3 and empty_count == 3:
            score += self.weights["my_1"]
        elif opp_count == self.window_size - 3 and empty_count == 3:
            score -= self.weights["opp_1"]

        return score

    def _score_center_control(self, board_obj, player_id, opp_id):
        """
        Cộng điểm kiểm soát trung tâm.

        Trong ConnectX, quân ở giữa tạo được nhiều đường thắng hơn quân ở biên,
        nên cùng là một quân nhưng quân ở cột giữa nên có giá trị cao hơn.
        """
        score = 0
        center = (board_obj.w - 1) / 2

        for c in range(board_obj.w):
            # Cột càng gần giữa thì weight càng cao.
            distance = abs(c - center)
            col_weight = int((board_obj.w - distance) * self.weights["center"])

            col_mask = 0
            for r in range(board_obj.h):
                col_mask |= 1 << (c * board_obj.col_height + r)

            my_count = (board_obj.boards[player_id] & col_mask).bit_count()
            opp_count = (board_obj.boards[opp_id] & col_mask).bit_count()
            score += (my_count - opp_count) * col_weight

        return score

    def _score_patterns_for_piece(self, piece_board, empty_board, playable_mask, board_obj, sign):
        """
        Chấm điểm tất cả window mở của một bên.

        sign = +1 nếu đang chấm quân mình
        sign = -1 nếu đang chấm quân đối thủ
        """
        score = 0

        for shift in board_obj.shifts:
            # Threat thắng ngay: x-1 quân + 1 ô trống có thể đánh ngay.
            immediate = self._count_pattern(
                piece_board=piece_board,
                empty_board=playable_mask,
                shift=shift,
                num_pieces=self.window_size - 1,
            )

            # Các window mở bình thường: x-1, x-2, x-3 quân và còn lại là ô trống.
            # Với Connect 4 tương ứng 3, 2, 1 quân.
            near_win = self._count_pattern(
                piece_board=piece_board,
                empty_board=empty_board,
                shift=shift,
                num_pieces=self.window_size - 1,
            )
            two = self._count_pattern(
                piece_board=piece_board,
                empty_board=empty_board,
                shift=shift,
                num_pieces=self.window_size - 2,
            )
            one = self._count_pattern(
                piece_board=piece_board,
                empty_board=empty_board,
                shift=shift,
                num_pieces=self.window_size - 3,
            ) if self.window_size >= 4 else 0

            if sign > 0:
                score += near_win * self.weights["my_3"]
                score += immediate * self.weights["my_immediate_win"]
                score += two * self.weights["my_2"]
                score += one * self.weights["my_1"]
            else:
                score -= near_win * self.weights["opp_3"]
                score -= immediate * self.weights["opp_immediate_win"]
                score -= two * self.weights["opp_2"]
                score -= one * self.weights["opp_1"]

        return score

    def evaluate(self, board_obj, player_id):
        """
        Trả về điểm heuristic của trạng thái board hiện tại.

        Hàm này KHÔNG đặt thử quân. Việc mô phỏng nước đi thuộc về ai.py/minimax.
        """
        opp_id = 1 - player_id

        # Terminal state: nếu trạng thái hiện tại đã thắng/thua thì trả điểm cực lớn.
        if board_obj.check_win(player_id):
            return self.WIN_SCORE
        if board_obj.check_win(opp_id):
            return self.LOSE_SCORE

        my_board = board_obj.boards[player_id]
        opp_board = board_obj.boards[opp_id]
        occupied = my_board | opp_board
        empty_board = (~occupied) & board_obj.valid_mask
        playable_mask = self._get_playable_mask(board_obj)

        score = 0

        # 1. Trung tâm: ưu tiên nhẹ, giúp khai cuộc/giữa game ổn định hơn.
        score += self._score_center_control(board_obj, player_id, opp_id)

        # 2. Pattern/threat của mình.
        score += self._score_patterns_for_piece(
            piece_board=my_board,
            empty_board=empty_board,
            playable_mask=playable_mask,
            board_obj=board_obj,
            sign=+1,
        )

        # 3. Pattern/threat của đối thủ.
        # Phạt nặng hơn để bot ưu tiên chặn nước thắng của đối phương.
        score += self._score_patterns_for_piece(
            piece_board=opp_board,
            empty_board=empty_board,
            playable_mask=playable_mask,
            board_obj=board_obj,
            sign=-1,
        )

        return score
