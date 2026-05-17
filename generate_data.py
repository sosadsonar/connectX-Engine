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

def worker_game(task_info):
    """
    HÀM WORKER CHẠY TRÊN TỪNG NHÂN CPU ĐỘC LẬP (Lock-free):
    Gánh trách nhiệm mô phỏng đúng 1 ván đấu đơn giữa 2 cấu hình não được chỉ định.
    Trả về danh sách lịch sử thế cờ và kết quả chung cuộc để luồng chính gom nhãn.
    """
    game_idx, w, h, x, weights_p0, weights_p1, depth, time_limit = task_info
    
    board = ConnectXBoard(w, h, x)
    # Khởi tạo 2 AI cục bộ độc lập hoàn toàn trên RAM của tiến trình con này
    ai0 = AdvancedNegamaxAI(weights_p0, player_id=0, tt_exponent=23)
    ai1 = AdvancedNegamaxAI(weights_p1, player_id=1, tt_exponent=23)
    
    ais = {0: ai0, 1: ai1}
    current_player = 0
    last_move = None
    game_history = []
    
    while board.get_valid_cols():
        active_ai = ais[current_player]
        
        # Ép xung tìm kiếm Negamax Alpha-Beta chặn verbose hiển thị để tránh nghẽn luồng I/O
        move = active_ai.select_move(board, max_depth=depth, time_limit=time_limit, last_move=last_move, verbose=False)
        if move == -1:
            break
            
        raw_score = getattr(active_ai, 'last_score', 0)
        
        # BỘ LỌC CHÍ MẠNG: Chỉ lấy trung cuộc chiến thuật, loại bỏ tuyệt đối sát cục cờ tàn hiển nhiên
        if abs(raw_score) <= 9000000:
            game_history.append({
                "us_mask": board.boards[current_player],
                "them_mask": board.boards[1 - current_player],
                "search_score": raw_score,
                "player_at_turn": current_player
            })
            
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
    parser.add_argument("--m1", type=str, default="models/best_weights_7x6_x4.json", help="Đường dẫn não vương quyền 7x6")
    parser.add_argument("--m2", type=str, default="models/best_weights_14x12_x4.json", help="Đường dẫn não vương quyền 14x12")
    parser.add_argument("--depth", type=int, default=12, help="Độ sâu duyệt cây khi sinh data")
    parser.add_argument("--time", type=float, default=1.8, help="Giới hạn thời gian nghĩ mỗi nước")
    parser.add_argument("--games", type=int, default=3000, help="Số ván đấu tự đối kháng muốn chạy")
    parser.add_argument("--cores", type=int, default=cpu_count(), help="Số nhân CPU muốn huy động (Mặc định: Toàn bộ máy)")
    parser.add_argument("--out", type=str, default="dataset_hybrid_nnue.npy", help="Tên file nhị phân đầu ra")
    args = parser.parse_args()

    print("=====================================================")
    print("    HỆ THỐNG SINH DỮ LIỆU ĐA NHÂN SIÊU TỐC (MULTIPROCESSING)")
    print(f" Sa bàn: {args.w}x{args.h} | Đang huy động: {args.cores}/{cpu_count()} nhân CPU 🔥")
    print("=====================================================\n")

    # Nạp tri thức nền từ 2 đại kiện tướng Heuristic
    weights_7x6 = load_champion_weights(args.m1)
    weights_14x12 = load_champion_weights(args.m2)
    
    # Phân rã cấu trúc tác vụ để đẩy xuống các nhân CPU
    print("[HỆ THỐNG] Đang thiết lập ma trận xoay tua tam phân chiến thuật...")
    tasks = []
    for idx in range(1, args.games + 1):
        if idx % 3 == 0:
            tasks.append((idx, args.w, args.h, args.x, weights_7x6, weights_7x6, args.depth, args.time))
        elif idx % 3 == 1:
            tasks.append((idx, args.w, args.h, args.x, weights_7x6, weights_14x12, args.depth, args.time))
        else:
            tasks.append((idx, args.w, args.h, args.x, weights_14x12, weights_7x6, args.depth, args.time))
            
    all_us_masks = []
    all_them_masks = []
    all_scores = []
    all_results = []
    
    total_positions_saved = 0
    game_count = 0
    start_time = time.time()

    print(f"[HỆ THỐNG] Khai hỏa! Đang xé nhỏ ván đấu giải phóng công suất phần cứng...\n")

    # KÍCH HOẠT POOL CÔNG NHÂN ĐA LUỒNG SÁT PHẦN CỨNG
    with Pool(processes=args.cores) as pool:
        # Sử dụng imap_unordered để luồng chính nhận kết quả ngay khi BẤT KỲ nhân nào làm xong
        for history, winner in pool.imap_unordered(worker_game, tasks, chunksize=2):
            game_count += 1
            
            # GÁN NHÃN KẾT QUẢ ĐỘNG [0, 0.5, 1] THEO GÓC NHÌN CHỦ THỂ (SIDE TO MOVE)
            for state in history:
                player_turn = state["player_at_turn"]
                
                if winner == -1:
                    result_label = 0.5  # Ván đấu Hòa
                elif winner == player_turn:
                    result_label = 1.0  # Phe đang cầm hình cờ này chung cuộc Thắng
                else:
                    result_label = 0.0  # Phe đang cầm hình cờ này chung cuộc Thua
                    
                all_us_masks.append(state["us_mask"])
                all_them_masks.append(state["them_mask"])
                all_scores.append(state["search_score"])
                all_results.append(result_label)
                
            total_positions_saved += len(history)
            
            # Báo cáo tiến độ động không nghẽn màn hình
            if game_count % 20 == 0 or game_count == 1 or game_count == args.games:
                elapsed = time.time() - start_time
                print(f" -> [Tiến độ: {game_count:04d}/{args.games:04d} ván] Tích lũy: {total_positions_saved:,} thế cờ sạch. Tốc độ thực: {game_count / elapsed:.2f} ván/giây.")

    # TỰ ĐỘNG KHÓA KIỂU DỮ LIỆU TRÁNH LỖI OVERFLOW KHI ĐỔI BÀN CỜ ĐẠI (15x15, 17x17)
    bit_required = args.w * (args.h + 1)
    mask_dtype = object if bit_required > 64 else np.uint64
    
    if mask_dtype == object:
        print(f"\n[HỆ THỐNG] Phát hiện hình cờ khổng lồ (>64-bit). Ép dải số nguyên lớn Python Object.")

    # Đóng gói ma trận nhị phân Numpy
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

    # Khạc trực tiếp file .npy dạng cấu trúc nén ra ổ cứng
    np.save(args.out, dataset_matrix)
    
    total_time = time.time() - start_time
    print(f"\n=====================================================")
    print("               CHIẾN DỊCH HOÀN TẤT MỸ MÃN")
    print("=====================================================")
    print(f"🏆 Tổng số Data Point tích lũy: {total_positions_saved:,} thế cờ trung cuộc.")
    print(f"💾 Tệp tin nhị phân xuất xưởng: '{args.out}'")
    print(f"⏱️ Tổng thời gian vắt kiệt CPU: {total_time/60:.2f} phút.")
    print(f"⚡ Hiệu suất trung bình: {total_positions_saved / total_time:.1f} thế cờ/giây.")
    print("=====================================================")

if __name__ == "__main__":
    # Bắt buộc phải có guard này để bảo vệ tiến trình con trên Windows/macOS không bị lặp vô hạn
    start_data_generation()
