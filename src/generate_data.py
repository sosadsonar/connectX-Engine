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
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    from train import BASE_CHAMPION_WEIGHTS
    print(f"[⚠️ WARNING] Không tìm thấy '{file_path}'. Fallback về cấu hình BASE.")
    return BASE_CHAMPION_WEIGHTS

def mirror_bitmask(mask: int) -> int:
    m = 0
    m |= (mask & 0x7F) << 42             
    m |= (mask & 0x3F80) << 28            
    m |= (mask & 0x1FC000) << 14          
    m |= (mask & 0xFF80000)               
    m |= (mask & 0x7F8000000) >> 14       
    m |= (mask & 0x3F800000000) >> 28     
    m |= (mask & 0x1FC0000000000) >> 42   
    return m

def save_dataset(base_file_path, data_dict, args, suffix):
    """Lưu file kèm hậu tố (ví dụ: _train.npy, _val.npy)"""
    file_path = base_file_path.replace(".npy", f"_{suffix}.npy")
    bit_required = args.w * (args.h + 1)
    mask_dtype = object if bit_required > 64 else np.uint64
    
    dataset_matrix = {
        "us_mask": np.array(data_dict["us"], dtype=mask_dtype),
        "them_mask": np.array(data_dict["them"], dtype=mask_dtype),
        "search_score": np.array(data_dict["scores"], dtype=np.int32),
        "game_result": np.array(data_dict["results"], dtype=np.float32),
        "meta_board": np.array([args.w, args.h, args.x], dtype=np.int32)
    }
    np.save(file_path, dataset_matrix)

def worker_game(task_info):
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
            
        # 🎲 ÉP 1 NƯỚC ĐẦU TIÊN RANDOM (Tránh lặp ván cờ)
        if ply_count == 0:
            move = random.choice(valid_cols)
        else:
            active_ai = ais[current_player]
            move = active_ai.select_move(board, max_depth=depth, time_limit=time_limit, last_move=last_move, verbose=False)
            if move == -1:
                break
                
            raw_score = getattr(active_ai, 'last_score', 0)
            
            # LỌC SÁT CỤC
            if abs(raw_score) > 9000000:
                previous_score = raw_score 
                board.make_move(move, current_player)
                last_move = move
                ply_count += 1
                continue 
                
            # LỌC BLUNDER 
            is_blunder_sequence = False
            if previous_score is not None:
                if abs(previous_score) <= 9000000:
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
    parser = argparse.ArgumentParser(description="Multi-Core Data Generator (Auto 8:2 Split)")
    parser.add_argument("--w", type=int, default=7, help="Chiều rộng sa bàn")
    parser.add_argument("--h", type=int, default=6, help="Chiều cao sa bàn")
    parser.add_argument("--x", type=int, default=4, help="Luật Connect X")
    parser.add_argument("--m1", type=str, default="../models/best_weights_7x6_x4.json", help="Đường dẫn HCE 7x6")
    parser.add_argument("--m2", type=str, default="../models/best_weights_14x12_x4.json", help="Đường dẫn HCE 14x12")
    parser.add_argument("--depth", type=int, default=12, help="Độ sâu duyệt cây")
    parser.add_argument("--time", type=float, default=1.8, help="Thời gian nghĩ")
    parser.add_argument("--games", type=int, default=25000, help="Số ván đấu thô muốn chạy")
    parser.add_argument("--cores", type=int, default=cpu_count(), help="Số nhân CPU muốn huy động")
    parser.add_argument("--out", type=str, default="../data/dataset_hybrid_nnue.npy", help="Base file name")
    parser.add_argument("--ckpt_interval", type=int, default=500, help="Chu kỳ lưu checkpoint")
    parser.add_argument("--blunder_thr", type=int, default=350000, help="Ngưỡng phát hiện Blunder")
    args = parser.parse_args()

    print("=====================================================")
    print("🔥 HỆ THỐNG SINH DATA KÉP (TỰ ĐỘNG CHIA 8:2 TRAIN/VAL)")
    print(f" Sa bàn: {args.w}x{args.h} | Đang huy động: {args.cores}/{cpu_count()} nhân CPU")
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
            
    # Tạo 2 túi chứa data riêng biệt
    train_data = {"us": [], "them": [], "scores": [], "results": []}
    val_data = {"us": [], "them": [], "scores": [], "results": []}
    
    game_count = 0
    start_time = time.time()

    with Pool(processes=args.cores) as pool:
        for history, winner in pool.imap_unordered(worker_game, tasks, chunksize=2):
            game_count += 1
            
            # 🎯 TUNG XÚC XẮC TỶ LỆ 80% TRAIN - 20% VAL CHO CẢ VÁN ĐẤU NÀY
            is_train = random.random() < 0.8
            target_dict = train_data if is_train else val_data
            
            for state in history:
                player_turn = state["player_at_turn"]
                if winner == -1: result_label = 0.5
                elif winner == player_turn: result_label = 1.0
                else: result_label = 0.0
                    
                # Data gốc
                target_dict["us"].append(state["us_mask"])
                target_dict["them"].append(state["them_mask"])
                target_dict["scores"].append(state["search_score"])
                target_dict["results"].append(result_label)
                
                # Data lật gương
                target_dict["us"].append(mirror_bitmask(state["us_mask"]))
                target_dict["them"].append(mirror_bitmask(state["them_mask"]))
                target_dict["scores"].append(state["search_score"]) 
                target_dict["results"].append(result_label)         
            
            if game_count % 20 == 0 or game_count == args.games:
                total_pos = len(train_data["us"]) + len(val_data["us"])
                elapsed = time.time() - start_time
                print(f" -> [Tiến độ: {game_count:05d}/{args.games:05d}] Tích lũy: {total_pos:,} thế cờ. Tốc độ: {game_count / elapsed:.2f} ván thô/giây.")

            # Lưu Checkpoint Kép
            if game_count % args.ckpt_interval == 0 and game_count < args.games:
                save_dataset(args.out, train_data, args, "train_ckpt")
                save_dataset(args.out, val_data, args, "val_ckpt")
                print(f"    🛡️ Đã lưu Checkpoint Train/Val an toàn.")

    # Xóa checkpoint thừa và lưu file cuối cùng
    ckpt_train = args.out.replace(".npy", "_train_ckpt.npy")
    ckpt_val = args.out.replace(".npy", "_val_ckpt.npy")
    if os.path.exists(ckpt_train): os.remove(ckpt_train)
    if os.path.exists(ckpt_val): os.remove(ckpt_val)

    save_dataset(args.out, train_data, args, "train")
    save_dataset(args.out, val_data, args, "val")
    
    total_time = time.time() - start_time
    print(f"\n=====================================================")
    print("        HOÀN TẤT SINH DỮ LIỆU & PHÂN TÁCH TỰ ĐỘNG")
    print("=====================================================")
    print(f"📦 Tập TRAIN (80%): {len(train_data['scores']):,} thế cờ -> {args.out.replace('.npy', '_train.npy')}")
    print(f"📦 Tập VAL   (20%): {len(val_data['scores']):,} thế cờ -> {args.out.replace('.npy', '_val.npy')}")
    print(f"⏱️ Tổng thời gian treo máy: {total_time/60:.2f} phút.")
    print("=====================================================")

if __name__ == "__main__":
    start_data_generation()