# ai.py
import time
import os
import numpy as np
from board import ConnectXBoard
from zobrist import TranspositionTable
from evaluator import BitboardCNNEvaluator
from debug import SearchDebugger
from move_order import MoveSorter  

EXACT = 0
LOWERBOUND = 1
UPPERBOUND = 2

class AdvancedNegamaxAI:
    def __init__(self, config_weights, player_id: int, mode: str = "heuristic", tt_exponent=23):
        """
        Khởi tạo AI hỗ trợ đa chế độ: 
        - "heuristic": Thuần công thức lọc tay cũ
        - "nnue": Thuần mạng nơ-ron thực chiến
        - "hybrid": Lò luyện lai mã CỔNG GÁC CHIẾN THUẬT (Gated Hybrid) 🛡️🔥
        """
        self.player_id = player_id
        self.mode = mode  # "heuristic", "nnue", "hybrid"
        self.tt = TranspositionTable(exponent=tt_exponent)
        self.sorter = None
        
        self.start_time = 0
        self.time_limit = 1.8  
        self.is_timeout = False
        self.node_count = 0

        # CẤU HÌNH TÀI NGUYÊN NỀN TẢNG THÍCH ỨNG
        if self.mode == "nnue":
            self.weights = {"WIN_BASE": 10000000, "ASPIRATION_DELTA": 3000}
        else:
            self.weights = config_weights
            self.evaluator = BitboardCNNEvaluator(config_weights)
            
        self.debugger = SearchDebugger(self.weights, player_id)

        # KÍCH HOẠT PHẦN CỨNG MẠNG NƠ-RON NẾU CHẠY CHẾ ĐỘ CÓ NNUE
        if self.mode in ["nnue", "hybrid"]:
            weights_path = "../nnue/nnue_weights (1).npz"
            if os.path.exists(weights_path):
                with np.load(weights_path) as data:
                    self.W1, self.b1 = data["W1"], data["b1"]
                    self.W2, self.b2 = data["W2"], data["b2"]
                    self.W3, self.b3 = data["W3"], data["b3"]
                print(f"🧠 [HỆ THỐNG] Đã kích hoạt Não NNUE NumPy cho chế độ: {self.mode.upper()} (Bot P{player_id})")
            else:
                raise FileNotFoundError(f"❌ [LỖI Chí Mạng] Không tìm thấy tệp trọng số '{weights_path}' trong repo!")
                
            self.shifts = np.array([c * 7 + r for c in range(7) for r in range(6)], dtype=np.uint64)
            self.input_buffer = np.zeros(84, dtype=np.float32)

    def check_signals(self):
        """Kiểm tra thời gian giới hạn tập trung"""
        self.node_count += 1
        if self.node_count & 1023 == 0:
            if time.time() - self.start_time > self.time_limit:
                self.is_timeout = True

    def evaluate_nnue(self, board: ConnectXBoard, current_player_id: int) -> int:
        """Hàm lượng giá NNUE Clipped ReLU tốc độ phần cứng bằng NumPy"""
        us_mask = board.boards[current_player_id]
        them_mask = board.boards[1 - current_player_id]
        
        self.input_buffer[:42] = (us_mask >> self.shifts) & 1
        self.input_buffer[42:] = (them_mask >> self.shifts) & 1
        
        h1 = np.clip(self.input_buffer @ self.W1 + self.b1, 0.0, 1.0)
        h2 = np.clip(h1 @ self.W2 + self.b2, 0.0, 1.0)
        output = h2 @ self.W3 + self.b3
        return int(output[0])

    def evaluate_hybrid(self, board: ConnectXBoard, current_player_id: int) -> int:
        """
        🛡️ VŨ KHÍ GATED HYBRID CHỐNG MÙ CHIẾN THUẬT:
        Hiện thực hóa chính xác 100% sơ đồ thiết kế bộ lọc cổng gác của ông.
        """
        # 1. Ép HCE chạy trước để dò quét bẫy rập hình học (Fork/Sát cục)
        hce_score = self.evaluator.evaluate(board, current_player_id)
        
        # ⚔️ 【 KỊCH BẢN A 】: Phát hiện biến động lớn (Ngưỡng bẫy Fork từ 500k điểm trở lên)
        if abs(hce_score) >= 500000:
            return hce_score # HCE giữ quyền phán quyết tuyệt đối, bỏ qua NNUE
            
        # 🕊️ 【 KỊCH BẢN B 】: Thế cờ lặng (Quiet Position)
        # Triệu hồi trực giác vĩ mô của mạng nơ-ron phối hợp xử lý vị trí
        nnue_score = self.evaluate_nnue(board, current_player_id)
        
        # Trộn điểm theo tỷ lệ vàng sơ đồ: 30% Chiến thuật HCE + 70% Vị trí chiến lược NNUE
        final_score = (0.3 * hce_score) + (0.7 * nnue_score)
        return int(final_score)

    def quiesce(self, board: ConnectXBoard, alpha: int, beta: int, current_player_id: int, ply: int) -> int:
        """Quiescence Search - Tìm kiếm tĩnh chặn đòn sát cục ngắn hạn"""
        self.check_signals()
        if self.is_timeout:
            return 0

        # RẼ NHÁNH ĐƯỜNG ỐNG LƯỢNG GIÁ TĨNH VỚI LÕI CỔNG GÁC MỚI
        if self.mode == "nnue":
            stand_pat = self.evaluate_nnue(board, current_player_id)
        elif self.mode == "hybrid":
            stand_pat = self.evaluate_hybrid(board, current_player_id)
        else:
            stand_pat = self.evaluator.evaluate(board, current_player_id)
            
        if stand_pat >= beta:
            return stand_pat
        alpha = max(alpha, stand_pat)

        valid_cols = board.get_valid_cols()
        opp_id = 1 - current_player_id

        for col in valid_cols:
            board.make_move(col, current_player_id)
            is_win = board.check_win(current_player_id)
            board.undo_move(col, current_player_id)
            if is_win:
                return self.weights["WIN_BASE"] - (ply + 1)

        forced_cols = []
        for col in valid_cols:
            board.make_move(col, opp_id)
            is_win = board.check_win(opp_id)
            board.undo_move(col, opp_id)
            if is_win:
                forced_cols.append(col)

        if len(forced_cols) > 1:
            return -self.weights["WIN_BASE"] + (ply + 2)

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

        opp_id = 1 - current_player_id
        if board.check_win(opp_id):
            return -self.weights["WIN_BASE"] + ply
        if board.check_win(current_player_id):
            return self.weights["WIN_BASE"] - ply

        valid_cols = board.get_valid_cols()
        if depth == 0 or not valid_cols:
            if self.mode == "nnue":
                return self.quiesce(board, alpha, beta, current_player_id, ply)
            elif self.mode == "hybrid":
                return self.quiesce(board, alpha, beta, current_player_id, ply)
            else:
                return self.quiesce(board, alpha, beta, current_player_id, ply)

        tt_move = tt_entry[4] if tt_entry else None
        ordered_moves = self.sorter.get_ordered_moves(
            board, valid_cols, tt_move, ply, 
            current_player_id=current_player_id, depth=depth
        )

        max_eval = float('-inf')
        best_move = ordered_moves[0] if ordered_moves else None

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
                if col != tt_move:  
                    if self.sorter.killer_moves[ply][0] != col:
                        self.sorter.killer_moves[ply][1] = self.sorter.killer_moves[ply][0]
                        self.sorter.killer_moves[ply][0] = col
                    self.sorter.history_table[col] += depth * depth
                break  

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

        for current_depth in range(1, max_depth + 1):
            if current_depth >= 3:
                alpha = last_depth_score - ASPIRATION_DELTA
                beta = last_depth_score + ASPIRATION_DELTA
            else:
                alpha = float('-inf')
                beta = float('inf')

            depth_timed_out = False

            tt_entry = self.tt.lookup(board.zobrist_key)
            best_move_suggestion = tt_entry[4] if tt_entry else None
            
            ordered_cols = self.sorter.get_ordered_moves(
                board, valid_cols, best_move_suggestion, ply=1, 
                current_player_id=self.player_id, last_move=last_move, depth=current_depth
            )

            while True:
                current_alpha = alpha
                current_beta = beta
                
                best_score = float('-inf')
                best_col = None
                scores_cache = {}
                
                for i, col in enumerate(ordered_cols):
                    board.make_move(col, self.player_id)
                    if i == 0:
                        score = -self.negamax(board, current_depth - 1, -current_beta, -current_alpha, 1 - self.player_id, ply=1)
                    else:
                        score = -self.negamax(board, current_depth - 1, -current_alpha - 1, -current_alpha, 1 - self.player_id, ply=1)
                        if current_alpha < score < current_beta and not self.is_timeout:
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

                if best_score <= alpha:
                    alpha = float('-inf')  
                    continue
                elif best_score >= beta:
                    beta = float('inf')    
                    continue
                break  

            if depth_timed_out:
                break
                
            if best_col is not None:
                overall_best_col = best_col
                last_depth_score = best_score  
                
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
                
        if verbose and best_completed_depth_info is not None:
            self.debugger.print_top_variations(
                deepest_depth=best_completed_depth_info["depth"],
                elapsed=best_completed_depth_info["elapsed"],
                node_count=best_completed_depth_info["node_count"],
                top_variations=best_completed_depth_info["variations"]
            )
            
        self.last_score = last_depth_score
        return overall_best_col