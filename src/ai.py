# ai.py
import time
from board import ConnectXBoard
from zobrist import TranspositionTable
from evaluator import BitboardCNNEvaluator
from debug import SearchDebugger
from move_order import MoveSorter  
from opening_book import PythonOpeningBookReader

EXACT = 0
LOWERBOUND = 1
UPPERBOUND = 2

class SearchTimeoutException(Exception):
    """Ngoại lệ tùy chỉnh dùng để cắt đứt ngay lập tức cây đệ quy khi hết giờ"""
    pass

class AdvancedNegamaxAI:
    def __init__(self, config_weights, player_id: int, mode: str = "heuristic", tt_exponent=23):
        self.player_id = player_id
        self.mode = "heuristic" # Ép buộc chạy chế độ heuristic thuần túy
        self.tt = TranspositionTable(exponent=tt_exponent)
        self.sorter = None
        
        self.start_time = 0
        self.time_limit = 2.2 
        self.is_timeout = False
        self.node_count = 0

        self.weights = config_weights
        self.evaluator = BitboardCNNEvaluator(config_weights)
        self.debugger = SearchDebugger(self.weights, player_id)
        self.native_book = PythonOpeningBookReader(book_path="../data/7x6.book")


    def check_signals(self):
        self.node_count += 1
        if self.node_count & 1023 == 0:
            if time.perf_counter() - self.start_time > self.time_limit:
                self.is_timeout = True
                raise SearchTimeoutException()

    def quiesce(self, board: ConnectXBoard, alpha: int, beta: int, current_player_id: int, ply: int) -> int:
        self.check_signals()

        stand_pat = self.evaluator.evaluate(board, current_player_id)
        if stand_pat >= beta:
            return stand_pat
        alpha = max(alpha, stand_pat)
        valid_cols = board.get_valid_cols()
        opp_id = 1 - current_player_id

        for col in valid_cols:
            board.make_move(col, current_player_id)
            try:
                is_win = board.check_win(current_player_id)
            finally:
                board.undo_move(col, current_player_id)
            if is_win:
                return self.weights["WIN_BASE"] - (ply + 1)

        forced_cols = []
        for col in valid_cols:
            board.make_move(col, opp_id)
            try:
                is_win = board.check_win(opp_id)
            finally:
                board.undo_move(col, opp_id)
            if is_win:
                forced_cols.append(col)

        if len(forced_cols) > 1:
            return -self.weights["WIN_BASE"] + (ply + 2)

        for col in forced_cols:
            board.make_move(col, current_player_id)
            try:
                score = -self.quiesce(board, -beta, -alpha, opp_id, ply + 1)
            finally:
                board.undo_move(col, current_player_id)

            if score >= beta:
                return score
            alpha = max(alpha, score)

        return alpha

    def negamax(self, board: ConnectXBoard, depth: int, alpha: int, beta: int, current_player_id: int, ply: int) -> int:
        alpha_orig = alpha
        self.check_signals()

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
            return self.quiesce(board, alpha, beta, current_player_id, ply)

        tt_move = tt_entry[4] if tt_entry else None
        
        # ĐỒNG BỘ VỊ TRÍ 1: Sửa hàm gọi trong Negamax đệ quy về hệ 3 tham số phẳng
        ordered_moves = self.sorter.get_ordered_moves(
            board, 
            tt_move=tt_move, 
            current_depth=depth
        )

        max_eval = float('-inf')
        best_move = ordered_moves[0] if ordered_moves else None

        for i, col in enumerate(ordered_moves):
            board.make_move(col, current_player_id)
            try:
                if i == 0:
                    score = -self.negamax(board, depth - 1, -beta, -alpha, opp_id, ply + 1)
                else:
                    score = -self.negamax(board, depth - 1, -alpha - 1, -alpha, opp_id, ply + 1)
                    if alpha < score < beta:
                        score = -self.negamax(board, depth - 1, -beta, -score, opp_id, ply + 1)
            finally:
                board.undo_move(col, current_player_id)

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

    def select_move(self, board: ConnectXBoard, max_depth=20, time_limit=2.2, last_move=None, verbose=True) -> int:
        if self.sorter is None:
            self.sorter = MoveSorter(board.w)

        valid_cols = board.get_valid_cols()
        if not valid_cols:
            return -1
        if len(valid_cols) == 1:
            return valid_cols[0]

        # 💥 SỬA ĐỔI CỐT LÕI: Thiết lập mốc thời gian NGAY LẬP TỨC để bảo vệ hàm check_signals
        self.start_time = time.perf_counter()
        self.time_limit = time_limit
        self.is_timeout = False
        self.node_count = 0

        # 🎯 ĐOẠN LOGIC TRA CỨU KHAI CUỘC LAI ĐỆ QUY AN TOÀN TUYỆT ĐỐI:
        if hasattr(self, 'native_book') and self.native_book.is_loaded:
            current_state_score = self.native_book.get_score_from_book(board, self.player_id)
            
            if current_state_score is not None:
                if verbose:
                    print(f"📖 [OPENING BOOK] Phát hiện thế cờ trong sách. Điểm mục tiêu của Bot: {current_state_score}")
                
                move_scores = []
                debug_info = {}
                
                # Bọc toàn bộ quá trình đệ quy phụ bằng try-except phòng ngừa cạn kiệt thời gian trên CPU yếu
                try:
                    for col in valid_cols:
                        board.make_move(col, self.player_id)
                        try:
                            book_score = self.native_book.get_score_from_book(board, 1 - self.player_id)
                            
                            if book_score is not None:
                                negamax_score = -book_score
                                debug_info[col] = f"{negamax_score} (Trích xuất từ Sách)"
                            else:
                                # Tự tính toán đệ quy ngắn hạn
                                raw_negamax = self.negamax(board, depth=4, alpha=float('-inf'), beta=float('inf'), current_player_id=1-self.player_id, ply=1)
                                negamax_score = -raw_negamax
                                debug_info[col] = f"{negamax_score} (CPU Negamax tự tính)"
                            
                            move_scores.append((negamax_score, col))
                        finally:
                            board.undo_move(col, self.player_id)
                except SearchTimeoutException:
                    if verbose:
                        print("⚠️ [OPENING BOOK] Đệ quy phụ bị ngắt quãng do hết thời gian khống chế!")
                
                if verbose and debug_info:
                    print("\n📊 [ANALYSIS] Kết quả phân tích tất cả các cột hợp lệ:")
                    print("-" * 65)
                    for c in sorted(valid_cols):
                        print(f"   • Cột [{c}]: Điểm số = {debug_info.get(c, 'None (Chưa tính kịp)')}")
                    print("-" * 65)
                
                if move_scores:
                    move_scores.sort(key=lambda item: (item[0], -abs(item[1] - (board.w // 2))), reverse=True)
                    best_score, best_col = move_scores[0]
                    if verbose:
                        print(f"📖 => Bot quyết định chọn cột tối ưu: [{best_col}] với điểm số: {best_score}\n")
                    return best_col

        # ──────────────────────────────────────────────────────────
        # Đoạn code tính toán Heuristic Chuyên sâu phía dưới giữ nguyên bản cũ, 
        # xóa bỏ các dòng gán trùng start_time, time_limit, node_count thừa đi:
        self.sorter.clear_history()

        overall_best_col = valid_cols[0]
        last_depth_score = 0  
        
        if verbose:
            self.debugger.print_start_turn(board)
        
        ASPIRATION_DELTA = int(self.weights.get("ASPIRATION_DELTA", 3000))
        best_completed_depth_info = None

        try:
            for current_depth in range(1, max_depth + 1):
                if current_depth >= 3:
                    alpha = last_depth_score - ASPIRATION_DELTA
                    beta = last_depth_score + ASPIRATION_DELTA
                else:
                    alpha = float('-inf')
                    beta = float('inf')

                tt_entry = self.tt.lookup(board.zobrist_key)
                best_move_suggestion = tt_entry[4] if tt_entry else None
                
                ordered_cols = self.sorter.get_ordered_moves(
                    board, 
                    tt_move=best_move_suggestion, 
                    current_depth=current_depth
                )

                while True:
                    current_alpha = alpha
                    current_beta = beta
                    
                    best_score = float('-inf')
                    best_col = None
                    scores_cache = {}
                    
                    for i, col in enumerate(ordered_cols):
                        board.make_move(col, self.player_id)
                        try:
                            if i == 0:
                                score = -self.negamax(board, current_depth - 1, -current_beta, -current_alpha, 1 - self.player_id, ply=1)
                            else:
                                score = -self.negamax(board, current_depth - 1, -current_alpha - 1, -current_alpha, 1 - self.player_id, ply=1)
                                if current_alpha < score < current_beta:
                                    score = -self.negamax(board, current_depth - 1, -current_beta, -score, 1 - self.player_id, ply=1)
                        finally:
                            board.undo_move(col, self.player_id)
                            
                        scores_cache[col] = score
                        if score > best_score:
                            best_score = score
                            best_col = col
                            
                        current_alpha = max(current_alpha, score)
                        if current_alpha >= current_beta:
                            break

                    if best_score <= alpha:
                        alpha = float('-inf')  
                        continue
                    elif best_score >= beta:
                        beta = float('inf')    
                        continue
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
                        "elapsed": time.perf_counter() - self.start_time,
                        "node_count": self.node_count,
                        "variations": top_variations
                    }

                if best_score >= self.weights["WIN_BASE"] - 100:
                    break
                    
        except SearchTimeoutException:
            pass
            
        if verbose and best_completed_depth_info is not None:
            self.debugger.print_top_variations(
                deepest_depth=best_completed_depth_info["depth"],
                elapsed=best_completed_depth_info["elapsed"],
                node_count=best_completed_depth_info["node_count"],
                top_variations=best_completed_depth_info["variations"]
            )
            if self.is_timeout:
                 print(f"⌛ [AI] Vừa ngắt tìm kiếm do chạm mốc thời gian {self.time_limit}s.")
            
        self.last_score = last_depth_score
        return overall_best_col