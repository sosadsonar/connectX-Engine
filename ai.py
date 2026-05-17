# ai.py
import time
from board import ConnectXBoard
from zobrist import TranspositionTable
from evaluator import BitboardCNNEvaluator

EXACT = 0
LOWERBOUND = 1
UPPERBOUND = 2

class AdvancedNegamaxAI:
    def __init__(self, config_weights, player_id: int, tt_exponent=21):
        """Khởi tạo AI với bộ trọng số, ID người chơi và bảng băm tương ứng"""
        self.weights = config_weights
        self.player_id = player_id
        self.evaluator = BitboardCNNEvaluator(config_weights)
        self.tt = TranspositionTable(exponent=tt_exponent)
        
        # Các tham số kiểm soát thời gian
        self.start_time = 0
        self.time_limit = 1.8  # Ngưỡng an toàn (Dưới 2.0s của luật đấu)
        self.is_timeout = False
        self.node_count = 0

    def negamax(self, board: ConnectXBoard, depth: int, alpha: int, beta: int, current_player_id: int) -> int:
        alpha_orig = alpha
        
        # Kiểm tra Timeout định kỳ sau mỗi 1024 nút để tiết kiệm chi phí gọi hàm time.time()
        self.node_count += 1
        if self.node_count & 1023 == 0:
            if time.time() - self.start_time > self.time_limit:
                self.is_timeout = True
        
        if self.is_timeout:
            return 0  # Trả về giá trị giả, kết quả tầng này sẽ bị hủy bỏ ở select_move

        # 1. Tra cứu Transposition Table từ bộ đệm Zobrist
        tt_entry = self.tt.lookup(board.zobrist_key)
        if tt_entry and tt_entry[1] >= depth:
            flag, score = tt_entry[2], tt_entry[3]
            if flag == EXACT:
                return score
            elif flag == LOWERBOUND:
                alpha = max(alpha, score)
            elif flag == UPPERBOUND:
                beta = min(beta, score)
            if alpha >= beta:
                return score

        # 2. Xử lý trạng thái kết thúc (Terminal Node) + Luật phân rã thời gian
        opp_id = 1 - current_player_id
        if board.check_win(opp_id):
            return -self.weights["WIN_BASE"] + depth
        if board.check_win(current_player_id):
            return self.weights["WIN_BASE"] - depth

        valid_cols = board.get_valid_cols()
        if depth == 0 or not valid_cols:
            return self.evaluator.evaluate(board, current_player_id)

        # 3. Move Ordering tối ưu tốc độ cắt tỉa
        best_move_suggestion = tt_entry[4] if tt_entry else None
        if best_move_suggestion in valid_cols:
            valid_cols.remove(best_move_suggestion)
            valid_cols.insert(0, best_move_suggestion)
        else:
            center = board.w // 2
            valid_cols.sort(key=lambda c: abs(c - center))

        max_eval = float('-inf')
        best_move = valid_cols[0] if valid_cols else None

        for col in valid_cols:
            board.make_move(col, current_player_id)
            score = -self.negamax(board, depth - 1, -beta, -alpha, 1 - current_player_id)
            board.undo_move(col, current_player_id)

            if self.is_timeout:
                return 0

            if score > max_eval:
                max_eval = score
                best_move = col
                
            alpha = max(alpha, score)
            if alpha >= beta:
                break

        # 4. Lưu kết quả thế cờ vào bảng băm vĩnh viễn (Chỉ lưu nếu chưa bị timeout)
        if not self.is_timeout:
            if max_eval <= alpha_orig:
                flag = UPPERBOUND
            elif max_eval >= beta:
                flag = LOWERBOUND
            else:
                flag = EXACT
            self.tt.store(board.zobrist_key, depth, flag, max_eval, best_move)
            
        return max_eval

    def select_move(self, board: ConnectXBoard, max_depth=20, time_limit=1.8) -> int:
        """Hàm giao tiếp chính sử dụng Iterative Deepening Search (IDS) để kiểm soát time < 2s"""
        valid_cols = board.get_valid_cols()
        if not valid_cols:
            return -1
        if len(valid_cols) == 1:
            return valid_cols[0]

        self.start_time = time.time()
        self.time_limit = time_limit
        self.is_timeout = False
        self.node_count = 0

        overall_best_col = valid_cols[0]
        
        # Vòng lặp đào sâu tăng dần (Iterative Deepening)
        for current_depth in range(1, max_depth + 1):
            best_score = float('-inf')
            best_col = None
            
            # Khởi tạo cửa sổ Alpha-Beta chuẩn xác (-inf đến inf) -> ĐÃ SỬA LỖI CHÍ MẠNG
            for col in valid_cols:
                board.make_move(col, self.player_id)
                score = -self.negamax(board, current_depth - 1, float('-inf'), float('inf'), 1 - self.player_id)
                board.undo_move(col, self.player_id)
                
                if self.is_timeout:
                    break
                    
                if score > best_score:
                    best_score = score
                    best_col = col
            
            # Nếu trong quá trình tính toán độ sâu này bị hết giờ, hủy kết quả của tầng này, lấy tầng trước
            if self.is_timeout:
                break
                
            if best_col is not None:
                overall_best_col = best_col
                
            # Sắp xếp lại valid_cols dựa trên kết quả tầng này để tầng sau cắt tỉa mạnh hơn nữa
            if best_col in valid_cols:
                valid_cols.remove(best_col)
                valid_cols.insert(0, best_col)

            # Nếu tìm thấy nước đi thắng tuyệt đối (Sát cục), dừng đào sâu để tiết kiệm thời gian
            if best_score >= self.weights["WIN_BASE"] - 100:
                break
                
        return overall_best_col