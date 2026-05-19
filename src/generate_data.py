# generate_data.py
import os
import time
import json
import random
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

def mirror_bitmask(mask: int) -> int:
    """
    VŨ KHÍ TỐI ƯU 1: Lật gương toàn bộ bàn cờ Trái <-> Phải bằng toán tử Bitwise.
    Khớp chính xác với cấu trúc Bitboard (c * 7 + r) 7x6 của Tromp, giúp nhân đôi dữ liệu siêu tốc.
    """
    m = 0
    m |= (mask & 0x7F) << 42              # Cột 0 -> Cột 6
    m |= (mask & 0x3F80) << 28            # Cột 1 -> Cột 5
    m |= (mask & 0x1FC000) << 14          # Cột 2 -> Cột 4
    m |= (mask & 0xFF80000)               # Cột 3 (Trung tâm) -> Giữ nguyên
    m |= (mask & 0x7F8000000) >> 14       # Cột 4 -> Cột 2
    m |= (mask & 0x3F800000000) >> 28     # Cột 5 -> Cột 1
    m |= (mask & 0x1FC0000000000) >> 42   # Cột 6 -> Cột 0
    return m

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
    
    previous_score = None 
    ply_count = 0 

    while board.get_valid_cols():
        valid_cols = board.get_valid_cols()
        if not valid_cols:
            break
            
        # 🎲 GIAI ĐOẠN 1: 6 nước đầu đi ngẫu nhiên hoàn toàn để tạo thế cờ dị
        if ply_count < 6:
            move = random.choice(valid_cols)
            
        # Từ nước thứ 7 trở đi, AI nghiêm túc vào cuộc và bắt đầu ghi log data sạch
        else:
            active_ai = ais[current_player]
            move = active_ai.select_move(board, max_depth=depth, time_limit=time_limit, last_move=last_move, verbose=False)
            if move == -1:
                break
                
            raw_score = getattr(active_ai, 'last_score', 0)
            
            # 1. BỘ LỌC SÁT CỤC: Loại bỏ trạng thái cờ tàn hiển nhiên (> 9,000,000)
            if abs(raw_score) <= 9000000:
                is_blunder_sequence = False
                
                # 2. BỘ LỌC BLUNDER (EVALUATION SWING)
                if previous_score is not None:
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
                    if game_history:
                        game_history.pop()
                
            previous_score = raw_score
            
        board.make_move(move, current_player)
        last_move = move
        ply_count += 1
        
        if board.check_win(current_player):
            return game_history, current_player
            
        current_player = 1 - current_player
        
    return game_history, -1

def start_data_generation():
    parser = argparse.ArgumentParser(description="ConnectX Production Multi-Core Data Generator with Symmetry Augmentation")
    parser.add_argument("--w", type=int, default=7, help="Chiều rộng sa bàn")
    parser.add_argument("--h", type=int, default=6, help="Chiều cao sa bàn")
    parser.add_argument("--x", type=int, default=4, help="Luật Connect X")
    parser.add_argument("--m1", type=str, default="../models/best_weights_7x6_x4.json", help="Đường dẫn não vương quyền 7x6")
    parser.add_argument("--m2", type=str, default="../models/best_weights_14x12_x4.json", help="Đường dẫn não vương quyền 14x12")
    parser.add_argument("--depth", type=int, default=12, help="Độ sâu duyệt cây khi sinh data")
    parser.add_argument("--time", type=float, default=1.8, help="Giới hạn thời gian nghĩ mỗi nước")
    parser.add_argument("--games", type=int, default=25000, help="Số ván đấu thô muốn chạy (Hệ thống tự x2 dữ liệu)")
    parser.add_argument("--cores", type=int, default=cpu_count(), help="Số nhân CPU muốn huy động")
    parser.add_argument("--out", type=str, default="../data/dataset_hybrid_nnue.npy", help="Tên file nhị phân đầu ra")
    parser.add_argument("--ckpt_interval", type=int, default=500, help="Chu kỳ lưu checkpoint dự phòng (số ván)")
    parser.add_argument("--blunder_thr", type=int, default=350000, help="Ngưỡng biến động điểm số để xác định Blunder")
    args = parser.parse_args()

    print("=====================================================")
    print("    🔥 HỆ THỐNG SINH DATA ĐA NHÂN KÍCH HOẠT PHẢN ỨNG LẬT GƯƠNG")
    print(f" Sa bàn: {args.w}x{args.h} | Đang huy động: {args.cores}/{cpu_count()} nhân CPU")
    print(f" Chế độ: Đột phá 6 nước đầu ngẫu nhiên + Tăng cường đối xứng gương 🧬")
    print(f" ⚠️  CHÚ Ý: Chạy {args.games:,} ván thô sẽ thu về tương đương {args.games * 2:,} ván data!")
    print("=====================================================\n")

    weights_7x6 = load_champion_weights(args.m1)
    weights_14x12 = load_champion_weights(args.m2)
    
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
                    
                # 🟢 BẢN GỐC TỪ THUẬT TOÁN TÌM KIẾM
                all_us_masks.append(state["us_mask"])
                all_them_masks.append(state["them_mask"])
                all_scores.append(state["search_score"])
                all_results.append(result_label)
                
                # 🧬 PHẢN ỨNG LẬT GƯƠNG: Nhân bản đối xứng hình học cấp tốc
                all_us_masks.append(mirror_bitmask(state["us_mask"]))
                all_them_masks.append(mirror_bitmask(state["them_mask"]))
                all_scores.append(state["search_score"]) # Điểm số lượng giá giữ nguyên tính chất đối xứng
                all_results.append(result_label)         # Kết quả ván đấu không đổi
                
            # Cập nhật bộ đếm x2 số lượng vị trí thực tế lưu vào RAM
            total_positions_saved += (len(history) * 2)
            
            if game_count % 20 == 0 or game_count == 1 or game_count == args.games:
                elapsed = time.time() - start_time
                print(f" -> [Tiến độ: {game_count:05d}/{args.games:05d} ván] Tích lũy: {total_positions_saved:,} thế cờ. Tốc độ thực: {game_count / elapsed:.2f} ván thô/giây.")

            if game_count % args.ckpt_interval == 0 and game_count < args.games:
                checkpoint_start = time.time()
                save_dataset(checkpoint_path, all_us_masks, all_them_masks, all_scores, all_results, args)
                ckpt_elapsed = time.time() - checkpoint_start
                print(f"    🛡️ [CHECKPOINT BẢO HIỂM X2] Đã lưu {total_positions_saved:,} thế cờ vào '{checkpoint_path}' (Mất {ckpt_elapsed:.2f}s).")

    save_dataset(args.out, all_us_masks, all_them_masks, all_scores, all_results, args)
    
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        
    total_time = time.time() - start_time
    print(f"\n=====================================================")
    print("               CHIẾN DỊCH HOÀN TẤT MỸ MÃN")
    print("=====================================================")
    print(f"🏆 Tổng số Data Point xuất xưởng: {total_positions_saved:,} thế cờ trung cuộc SIÊU PHẲNG.")
    print(f"💾 Tệp tin nhị phân lưu trữ: '{args.out}'")
    print(f"⏱️ Tổng thời gian treo máy: {total_time/60:.2f} phút.")
    print(f"⚡ Hiệu suất thực tế: {total_positions_saved / total_time:.1f} thế cờ/giây.")
    print("=====================================================")

if __name__ == "__main__":
    start_data_generation()