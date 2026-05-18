# generate_data.py
import os
import time
import json
import argparse
import numpy as np
from multiprocessing import Pool, cpu_count
from board import ConnectXBoard
from ai import AdvancedNegamaxAI

def load_champion_weights(file_path):
    """Nạp bộ trọng số vô địch làm đầu não tự đấu sinh dữ liệu"""
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    from train import BASE_CHAMPION_WEIGHTS
    print(f"[⚠️ WARNING] Không tìm thấy '{file_path}'. Fallback về cấu hình BASE.")
    return BASE_CHAMPION_WEIGHTS

def save_dataset(file_path, all_us_masks, all_them_masks, all_scores, all_results, args):
    """Hàm đóng gói dữ liệu và ghi xuống ổ cứng (Dùng cho cả checkpoint và kết quả cuối)"""
    bit_required = args.w * (args.h + 1)
    mask_dtype = object if bit_required > 64 else np.uint64
    
    np_us = np.array(all_us_masks, dtype=mask_dtype)
    np_them = np.array(all_them_masks, dtype=mask_dtype)
    np_scores = np.array(all_scores, dtype=np.int32)
    np_results = np.array(all_results, dtype=np.float32)

    dataset_matrix = {
        "us_mask": np_us,
        "them_mask": np_them,
        "search_score": np_scores,
        "game_result": np_results,
        "meta_board": np.array([args.w, args.h, args.x], dtype=np.int32)
    }
    np.save(file_path, dataset_matrix)

def worker_game(task_info):
    """
    HÀM WORKER CHẠY TRÊN TỪNG NHÂN CPU ĐỘC LẬP (Lock-free):
    Mô phỏng 1 ván đấu đơn, tích hợp bộ lọc Sát cục và bộ lọc Blunder siêu phẳng.
    """
    game_idx, w, h, x, weights_p0, weights_p1, depth, time_limit, blunder_thr = task_info
    
    board = ConnectXBoard(w, h, x)
    ai0 = AdvancedNegamaxAI(weights_p0, player_id=0, tt_exponent=23)
    ai1 = AdvancedNegamaxAI(weights_p1, player_id=1, tt_exponent=23)
    
    ais = {0: ai0, 1: ai1}
    current_player = 0
    last_move = None
    game_history = []
    
    # Biến theo dõi điểm số của nước đi ngay trước đó để tính độ lệch (Swing)
    previous_score = None 

    while board.get_valid_cols():
        active_ai = ais[current_player]
        
        move = active_ai.select_move(board, max_depth=depth, time_limit=time_limit, last_move=last_move, verbose=False)
        if move == -1:
            break
            
        # Lấy điểm số từ thuộc tính được cập nhật sau khi AI nghĩ xong
        raw_score = getattr(active_ai, 'last_score', 0)
        
        # 1. BỘ LỌC SÁT CỤC: Loại bỏ tuyệt đối trạng thái cờ tàn hiển nhiên (> 9,000,000)
        if abs(raw_score) <= 9000000:
            is_blunder_sequence = False
            
            # 2. BỘ LỌC BLUNDER (EVALUATION SWING): Kiểm tra độ lệch pha chiến thuật
            if previous_score is not None:
                # Theo nguyên lý zero-sum của Negamax, điểm kỳ vọng lượt này phải bằng -(điểm lượt trước)
                # Nếu độ lệch tuyệt đối vượt ngưỡng, chứng tỏ thế trận bị gãy do sai lầm nghiêm trọng hoặc hiệu ứng chân trời
                eval_swing = abs(raw_score - (-previous_score))
                if eval_swing > blunder_thr:
                    is_blunder_sequence = True
            
            if not is_blunder_sequence:
                game_history.append({
                    "us_mask": board.boards[current_player],
                    "them_mask": board.boards[1 - current_player],
                    "search_score": raw_score,
                    "player_at_turn": current_player
                })
            else:
                # Nếu phát hiện pha bẻ gãy điểm số đột ngột, ta tiến hành "hồi tố":
                # Xóa luôn thế cờ lỗi liền trước của đối thủ ra khỏi lịch sử để giữ tập dữ liệu siêu sạch
                if game_history:
                    game_history.pop()
            
        # Cập nhật điểm mốc để làm tham chiếu so sánh cho lượt đi kế tiếp
        previous_score = raw_score
        
        board.make_move(move, current_player)
        last_move = move
        
        if board.check_win(current_player):
            return game_history, current_player
            
        current_player = 1 - current_player
        
    return game_history, -1

def start_data_generation():
    parser = argparse.ArgumentParser(description="ConnectX Production Multi-Core Data Generator for NNUE")
    parser.add_argument("--w", type=int, default=7, help="Chiều rộng sa bàn")
    parser.add_argument("--h", type=int, default=6, help="Chiều cao sa bàn")
    parser.add_argument("--x", type=int, default=4, help="Luật Connect X")
    parser.add_argument("--m1", type=str, default="../models/best_weights_7x6_x4.json", help="Đường dẫn não vương quyền 7x6")
    parser.add_argument("--m2", type=str, default="../models/best_weights_14x12_x4.json", help="Đường dẫn não vương quyền 14x12")
    parser.add_argument("--depth", type=int, default=12, help="Độ sâu duyệt cây khi sinh data")
    parser.add_argument("--time", type=float, default=1.8, help="Giới hạn thời gian nghĩ mỗi nước")
    parser.add_argument("--games", type=int, default=3000, help="Số ván đấu tự đối kháng muốn chạy")
    parser.add_argument("--cores", type=int, default=cpu_count(), help="Số nhân CPU muốn huy động")
    parser.add_argument("--out", type=str, default="../data/dataset_hybrid_nnue.npy", help="Tên file nhị phân đầu ra")
    parser.add_argument("--ckpt_interval", type=int, default=500, help="Chu kỳ lưu checkpoint dự phòng (số ván)")
    # Bổ sung tham số cấu hình ngưỡng Blunder linh hoạt từ CLI
    parser.add_argument("--blunder_thr", type=int, default=350000, help="Ngưỡng biến động điểm số để xác định Blunder")
    args = parser.parse_args()

    print("=====================================================")
    print("    HỆ THỐNG SINH DỮ LIỆU ĐA NHÂN SIÊU TỐC (MULTIPROCESSING)")
    print(f" Sa bàn: {args.w}x{args.h} | Đang huy động: {args.cores}/{cpu_count()} nhân CPU 🔥")
    print(f" Ngưỡng lọc Blunder chiến thuật: {args.blunder_thr:,} điểm 🎯")
    print(f" Tự động lưu bảo hiểm sau mỗi: {args.ckpt_interval} ván cờ 🛡️")
    print("=====================================================\n")

    weights_7x6 = load_champion_weights(args.m1)
    weights_14x12 = load_champion_weights(args.m2)
    
    print("[HỆ THỐNG] Đang thiết lập ma trận xoay tua tam phân chiến thuật...")
    tasks = []
    for idx in range(1, args.games + 1):
        if idx % 3 == 0:
            tasks.append((idx, args.w, args.h, args.x, weights_7x6, weights_7x6, args.depth, args.time, args.blunder_thr))
        elif idx % 3 == 1:
            tasks.append((idx, args.w, args.h, args.x, weights_7x6, weights_14x12, args.depth, args.time, args.blunder_thr))
        else:
            tasks.append((idx, args.w, args.h, args.x, weights_14x12, weights_7x6, args.depth, args.time, args.blunder_thr))
            
    all_us_masks = []
    all_them_masks = []
    all_scores = []
    all_results = []
    
    total_positions_saved = 0
    game_count = 0
    start_time = time.time()

    print(f"[HỆ THỐNG] Khai hỏa! Đang xé nhỏ ván đấu giải phóng công suất phần cứng...\n")
    checkpoint_path = f"{args.out}.ckpt"

    with Pool(processes=args.cores) as pool:
        for history, winner in pool.imap_unordered(worker_game, tasks, chunksize=2):
            game_count += 1
            
            for state in history:
                player_turn = state["player_at_turn"]
                if winner == -1:
                    result_label = 0.5
                elif winner == player_turn:
                    result_label = 1.0
                else:
                    result_label = 0.0
                    
                all_us_masks.append(state["us_mask"])
                all_them_masks.append(state["them_mask"])
                all_scores.append(state["search_score"])
                all_results.append(result_label)
                
            total_positions_saved += len(history)
            
            if game_count % 20 == 0 or game_count == 1 or game_count == args.games:
                elapsed = time.time() - start_time
                print(f" -> [Tiến độ: {game_count:05d}/{args.games:05d} ván] Tích lũy: {total_positions_saved:,} thế cờ sạch. Tốc độ thực: {game_count / elapsed:.2f} ván/giây.")

            if game_count % args.ckpt_interval == 0 and game_count < args.games:
                checkpoint_start = time.time()
                save_dataset(checkpoint_path, all_us_masks, all_them_masks, all_scores, all_results, args)
                ckpt_elapsed = time.time() - checkpoint_start
                print(f"    💾 [CHECKPOINT] Đã ghi đè bảo hiểm tại ván {game_count}! Tích lũy {total_positions_saved:,} thế cờ sạch vào '{checkpoint_path}' (Mất {ckpt_elapsed:.2f}s).")

    bit_required = args.w * (args.h + 1)
    if bit_required > 64:
        print(f"\n[HỆ THỐNG] Phát hiện hình cờ khổng lồ (>64-bit). Ép dải số nguyên lớn Python Object.")

    save_dataset(args.out, all_us_masks, all_them_masks, all_scores, all_results, args)
    
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        
    total_time = time.time() - start_time
    print(f"\n=====================================================")
    print("               CHIẾN DỊCH HOÀN TẤT MỸ MÃN")
    print("=====================================================")
    print(f"🏆 Tổng số Data Point tích lũy: {total_positions_saved:,} thế cờ trung cuộc SIÊU SẠCH.")
    print(f"💾 Tệp tin nhị phân xuất xưởng: '{args.out}'")
    print(f"⏱️ Tổng thời gian vắt kiệt CPU: {total_time/60:.2f} phút.")
    print(f"⚡ Hiệu suất trung bình: {total_positions_saved / total_time:.1f} thế cờ/giây.")
    print("=====================================================")

if __name__ == "__main__":
    start_data_generation()