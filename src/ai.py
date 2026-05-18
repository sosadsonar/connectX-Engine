# ai.py
import time
from board import ConnectXBoard
from zobrist import TranspositionTable
from evaluator import BitboardCNNEvaluator
from debug import SearchDebugger
from move_order import MoveSorter  # Import bộ sorter mới tách

EXACT = 0
LOWERBOUND = 1
UPPERBOUND = 2

class AdvancedNegamaxAI:
    def __init__(self, config_weights, player_id: int, tt_exponent=23):
        """Khởi tạo AI với bộ não PVS + IDS + Aspiration Windows + Quiescence Search chuẩn hóa Ply"""
        self.weights = config_weights
        self.player_id = player_id
        self.evaluator = BitboardCNNEvaluator(config_weights)
        self.tt = TranspositionTable(exponent=tt_exponent)
        self.debugger = SearchDebugger(config_weights, player_id)
        
        # Bộ sắp xếp move sẽ được khởi tạo lười (lazy init) khi biết kích thước board
        self.sorter = None
        
        self.start_time = 0
        self.time_limit = 1.8  
        self.is_timeout = False
        self.node_count = 0

    def check_signals(self):
        """Kiểm tra thời gian giới hạn tập trung để tránh overhead gọi time.time() liên tục"""
        self.node_count += 1
        if self.node_count & 1023 == 0:
            if time.time() - self.start_time > self.time_limit:
                self.is_timeout = True

    def quiesce(self, board: ConnectXBoard, alpha: int, beta: int, current_player_id: int, ply: int) -> int:
        """Quiescence Search chống hiệu ứng chân trời kết hợp tính toán khoảng cách sát cục chuẩn xác"""
        self.check_signals()
        if self.is_timeout:
            return 0

        # Stand Pat: Lấy điểm lượng giá tĩnh làm mỏ neo nền
        stand_pat = self.evaluator.evaluate(board, current_player_id)
        if stand_pat >= beta:
            return stand_pat
        alpha = max(alpha, stand_pat)

        valid_cols = board.get_valid_cols()
        opp_id = 1 - current_player_id

        # 1. TẠO ĐÒN PHẢN CÔNG: Nếu mình có nước thắng ngay, chớp thời cơ lập tức
        for col in valid_cols:
            board.make_move(col, current_player_id)
            is_win = board.check_win(current_player_id)
            board.undo_move(col, current_player_id)
            if is_win:
                return self.weights["WIN_BASE"] - ply

        # 2. ĐÁNH CHẶN BUỘC THẾ: Tìm các cột đối thủ có thể sát cục vào lượt sau
        forced_cols = []
        for col in valid_cols:
            board.make_move(col, opp_id)
            is_win = board.check_win(opp_id)
            board.undo_move(col, opp_id)
            if is_win:
                forced_cols.append(col)

        # Nếu đối thủ có từ 2 nước sát cục độc lập trở lên -> Thua chắc chắn
        if len(forced_cols) > 1:
            return -self.weights["WIN_BASE"] + ply

        # Duyệt qua các nước đi ép buộc chặn đứng hiểm họa
        for col in forced_cols:
            board.make_move(col, current_player_id)
            score = -self.quiesce(board, -beta, -alpha, opp_id, ply + 1)
            board.undo_move(col, current_player_id)

            if self.is_timeout:
                return 0

            if score >= beta:
                return score
            alpha = max(alpha, score)

        return alpha

    def negamax(self, board: ConnectXBoard, depth: int, alpha: int, beta: int, current_player_id: int, ply: int) -> int:
        alpha_orig = alpha
        self.check_signals()
        if self.is_timeout:
            return 0  

        # 1. Tra cứu Transposition Table & Giải nén điểm sát cục động
        tt_entry = self.tt.lookup(board.zobrist_key)
        if tt_entry and tt_entry[1] >= depth:
            flag, score = tt_entry[2], tt_entry[3]
            
            if score > 9000000:
                score -= ply
            elif score < -9000000:
                score += ply
                
            if flag == EXACT:
                return score
            elif flag == LOWERBOUND:
                alpha = max(alpha, score)
            elif flag == UPPERBOUND:
                beta = min(beta, score)
            if alpha >= beta:
                return score

        # 2. Xử lý trạng thái kết thúc (Terminal Node)
        opp_id = 1 - current_player_id
        if board.check_win(opp_id):
            return -self.weights["WIN_BASE"] + ply
        if board.check_win(current_player_id):
            return self.weights["WIN_BASE"] - ply

        valid_cols = board.get_valid_cols()
        if depth == 0 or not valid_cols:
            return self.quiesce(board, alpha, beta, current_player_id, ply)

        # 3. ĐỒNG BỘ: Truyền đầy đủ current_player_id và depth vào bộ Sorter để đồng bộ logic
        tt_move = tt_entry[4] if tt_entry else None
        ordered_moves = self.sorter.get_ordered_moves(
            board, valid_cols, tt_move, ply, 
            current_player_id=current_player_id, depth=depth
        )

        max_eval = float('-inf')
        best_move = ordered_moves[0] if ordered_moves else None

        # 4. Lõi tìm kiếm biến thể chính (PVS)
        for i, col in enumerate(ordered_moves):
            board.make_move(col, current_player_id)
            if i == 0:
                score = -self.negamax(board, depth - 1, -beta, -alpha, opp_id, ply + 1)
            else:
                score = -self.negamax(board, depth - 1, -alpha - 1, -alpha, opp_id, ply + 1)
                if alpha < score < beta and not self.is_timeout:
                    score = -self.negamax(board, depth - 1, -beta, -score, opp_id, ply + 1)
            board.undo_move(col, current_player_id)

            if self.is_timeout:
                return 0

            if score > max_eval:
                max_eval = score
                best_move = col
                
            alpha = max(alpha, score)
            if alpha >= beta:
                # BETA-CUTOFF: Nước đi quá tốt, cập nhật ngay bộ nhớ Heuristics kì cựu
                if col != tt_move:  # Không lưu trùng nếu nó đã là nước đi tốt từ TT
                    # Đẩy nước Killer cũ xuống slot 2, lưu nước mới vào slot 1
                    if self.sorter.killer_moves[ply][0] != col:
                        self.sorter.killer_moves[ply][1] = self.sorter.killer_moves[ply][0]
                        self.sorter.killer_moves[ply][0] = col
                    # Cộng điểm History tỉ lệ thuận với bình phương độ sâu nhánh cờ
                    self.sorter.history_table[col] += depth * depth
                break  

        # 5. Lưu trữ kết quả vào bảng băm (TT Store)
        if not self.is_timeout:
            if max_eval <= alpha_orig:
                flag = UPPERBOUND
            elif max_eval >= beta:
                flag = LOWERBOUND
            else:
                flag = EXACT
                
            tt_score = max_eval
            if max_eval > 9000000:
                tt_score += ply
            elif max_eval < -9000000:
                tt_score -= ply
                
            self.tt.store(board.zobrist_key, depth, flag, tt_score, best_move)
            
        return max_eval

    def _extract_pv(self, board: ConnectXBoard, first_move: int, max_pv_depth: int) -> str:
        """Truy vết chuỗi nước đi lý tưởng (PV Line) từ Transposition Table"""
        pv = [first_move]
        board.make_move(first_move, self.player_id)
        states = [(first_move, self.player_id)]
        
        curr_player = 1 - self.player_id
        for _ in range(max_pv_depth - 1):
            tt_entry = self.tt.lookup(board.zobrist_key)
            if tt_entry and tt_entry[4] is not None:
                next_move = tt_entry[4]
                if next_move in board.get_valid_cols():
                    board.make_move(next_move, curr_player)
                    states.append((next_move, curr_player))
                    pv.append(next_move)
                    curr_player = 1 - curr_player
                else:
                    break
            else:
                break
                
        for m, p in reversed(states):
            board.undo_move(m, p)
            
        return " -> ".join(map(str, pv))

    def select_move(self, board: ConnectXBoard, max_depth=20, time_limit=1.8, last_move=None, verbose=True) -> int:
        if self.sorter is None:
            self.sorter = MoveSorter(board.w)

        valid_cols = board.get_valid_cols()
        if not valid_cols:
            return -1
        if len(valid_cols) == 1:
            return valid_cols[0]

        # Khởi tạo tài nguyên hệ thống
        self.start_time = time.time()
        self.time_limit = time_limit
        self.is_timeout = False
        self.node_count = 0
        self.sorter.clear_history()

        overall_best_col = valid_cols[0]
        last_depth_score = 0  
        
        if verbose:
            self.debugger.print_start_turn(board)
        
        ASPIRATION_DELTA = int(self.weights.get("ASPIRATION_DELTA", 3000))
        best_completed_depth_info = None

        # --- LOOP 1: ITERATIVE DEEPENING (Tăng dần độ sâu) ---
        for current_depth in range(1, max_depth + 1):
            
            # Thiết lập cửa sổ Aspiration Window ban đầu cho độ sâu này
            if current_depth >= 3:
                alpha = last_depth_score - ASPIRATION_DELTA
                beta = last_depth_score + ASPIRATION_DELTA
            else:
                alpha = float('-inf')
                beta = float('inf')

            depth_timed_out = False

            # TỐI ƯU: Đưa việc sinh và sắp xếp nước đi ở Root ra ngoài vòng lặp Re-search.
            tt_entry = self.tt.lookup(board.zobrist_key)
            best_move_suggestion = tt_entry[4] if tt_entry else None
            
            # ĐỒNG BỘ: Truyền đầy đủ thông tin ID của mình và độ sâu hiện tại vào Sorter tại Root
            ordered_cols = self.sorter.get_ordered_moves(
                board, valid_cols, best_move_suggestion, ply=1, 
                current_player_id=self.player_id, last_move=last_move, depth=current_depth
            )

            # --- LOOP 2: ASPIRATION RE-SEARCH (Tìm lại nếu vỡ cửa sổ) ---
            while True:
                current_alpha = alpha
                current_beta = beta
                
                best_score = float('-inf')
                best_col = None
                scores_cache = {}
                
                # --- LOOP 3: ROOT MOVES ITERATION (Duyệt các cột tại gốc) ---
                for i, col in enumerate(ordered_cols):
                    board.make_move(col, self.player_id)
                    
                    if i == 0:
                        # Nước đi kỳ vọng tốt nhất (PV Move): Duyệt với toàn bộ cửa sổ hiện tại
                        score = -self.negamax(board, current_depth - 1, -current_beta, -current_alpha, 1 - self.player_id, ply=1)
                    else:
                        # Các nước đi phía sau: Thử nghiệm với cửa sổ hẹp (Null Window)
                        score = -self.negamax(board, current_depth - 1, -current_alpha - 1, -current_alpha, 1 - self.player_id, ply=1)
                        if current_alpha < score < current_beta and not self.is_timeout:
                            # Nếu nước cờ thử nghiệm tốt bất thường -> Nghiên cứu lại với cửa sổ chuẩn
                            score = -self.negamax(board, current_depth - 1, -current_beta, -score, 1 - self.player_id, ply=1)
                            
                    board.undo_move(col, self.player_id)
                    
                    if self.is_timeout:
                        depth_timed_out = True
                        break
                        
                    scores_cache[col] = score
                    if score > best_score:
                        best_score = score
                        best_col = col
                        
                    current_alpha = max(current_alpha, score)
                    if current_alpha >= current_beta:
                        break

                if self.is_timeout or depth_timed_out:
                    depth_timed_out = True
                    break

                # Kiểm tra xem kết quả tìm kiếm có bị tràn ra ngoài biên Aspiration Window không
                if best_score <= alpha:
                    alpha = float('-inf')  # Thất bại thảm hại (Fail-Low)
                    continue
                elif best_score >= beta:
                    beta = float('inf')    # Đột biến bất ngờ (Fail-High)
                    continue
                
                break  # Cửa sổ bọc chuẩn xác, thoát Loop 2 để lên độ sâu tiếp theo

            if depth_timed_out:
                break
                
            # Lưu lại thành quả của độ sâu vừa hoàn thành trọn vẹn
            if best_col is not None:
                overall_best_col = best_col
                last_depth_score = best_score  
                
                # Trích xuất dữ liệu phục vụ Debug hiển thị PV Line
                sorted_cols = sorted(scores_cache.keys(), key=lambda c: scores_cache[c], reverse=True)
                top_variations = []
                for c in sorted_cols[:3]:
                    pv_str = self._extract_pv(board, c, current_depth)
                    top_variations.append((c, scores_cache[c], pv_str))
                
                best_completed_depth_info = {
                    "depth": current_depth,
                    "elapsed": time.time() - self.start_time,
                    "node_count": self.node_count,
                    "variations": top_variations
                }

            if best_score >= self.weights["WIN_BASE"] - 100:
                break
                
        # In kết quả báo cáo của độ sâu hoàn thiện nhất
        if verbose and best_completed_depth_info is not None:
            self.debugger.print_top_variations(
                deepest_depth=best_completed_depth_info["depth"],
                elapsed=best_completed_depth_info["elapsed"],
                node_count=best_completed_depth_info["node_count"],
                top_variations=best_completed_depth_info["variations"]
            )
            
        # GENERATE_DATA ĐỌC ĐƯỢC ĐIỂM LƯỢNG GIÁ:
        self.last_score = last_depth_score
                
        return overall_best_col