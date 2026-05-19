# sanitize_and_merge.py
import os
import random
import numpy as np

# =====================================================================
# ⚙️ CẤU HÌNH ĐƯỜNG DẪN FILE CỦA ÔNG VÀ BẠN ÔNG TẠI ĐÂY
# =====================================================================
FILES_NO_MIRROR = [
    "7x6x4-2500.npy", 
    "7x6x4-4000.npy"
]

FILES_WITH_MIRROR = [
    "7x6x4-8000.npy"  # File dở dang dính lật gương của ông
]

OUTPUT_TRAIN = "final_nnue_train.npy"
OUTPUT_VAL = "final_nnue_val.npy"

# Đặt Seed cố định để đảm bảo tính nhất quán khi chia tách ngẫu nhiên
random.seed(42)

def count_bits(n):
    """Tối ưu tốc độ bằng hàm đếm bit tầng C của Python"""
    return int(n).bit_count()

def extract_games_from_file(file_path, is_mirrored):
    """Trích xuất mảng phẳng thành danh sách các ván đấu độc lập cấu trúc nguyên vẹn"""
    if not os.path.exists(file_path):
        print(f"[⚠️ WARNING] Không tìm thấy file: {file_path}, bỏ qua.")
        return [], None
        
    print(f"📦 Đang bóc tách dữ liệu từ file: '{file_path}'...")
    data = np.load(file_path, allow_pickle=True).item()
    
    us_mask = data["us_mask"]
    them_mask = data["them_mask"]
    search_score = data["search_score"]
    game_result = data["game_result"]
    meta_board = data.get("meta_board", np.array([7, 6, 4], dtype=np.int32))
    
    num_samples = len(search_score)
    if num_samples == 0:
        return [], meta_board
        
    # Tính toán mảng quân số nền tảng để dò ranh giới
    print("  -> Đang quét tọa độ bitmask xác định mật độ quân số...")
    piece_counts = np.array([count_bits(u) + count_bits(t) for u, t in zip(us_mask, them_mask)])
    
    games = []
    start_idx = 0
    
    for i in range(1, num_samples):
        is_boundary = False
        if not is_mirrored:
            # File thô: Quân số giảm hoặc bằng nước trước = Ván mới khai cuộc
            if piece_counts[i] <= piece_counts[i-1]:
                is_boundary = True
        else:
            # File gương: Ranh giới chỉ xuất hiện ở chỉ mục chẵn khi cặp nước đi reset
            if i % 2 == 0 and piece_counts[i] <= piece_counts[i-1]:
                is_boundary = True
                
        if is_boundary:
            games.append({
                "us_mask": us_mask[start_idx:i],
                "them_mask": them_mask[start_idx:i],
                "search_score": search_score[start_idx:i],
                "game_result": game_result[start_idx:i]
            })
            start_idx = i
            
    # Gói nốt ván cờ cuối cùng của file
    games.append({
        "us_mask": us_mask[start_idx:],
        "them_mask": them_mask[start_idx:],
        "search_score": search_score[start_idx:],
        "game_result": game_result[start_idx:]
    })
    
    print(f"  -> Thành công! Trích xuất được {len(games):,} ván đấu hoàn chỉnh.")
    return games, meta_board

def main():
    all_games = []
    final_meta_board = np.array([7, 6, 4], dtype=np.int32)
    
    # 1. Thu thập từ nguồn chưa lật gương
    for fp in FILES_NO_MIRROR:
        games, meta = extract_games_from_file(fp, is_mirrored=False)
        all_games.extend(games)
        if meta is not None: final_meta_board = meta
        
    # 2. Thu thập từ nguồn đã dính lật gương
    for fp in FILES_WITH_MIRROR:
        games, meta = extract_games_from_file(fp, is_mirrored=True)
        all_games.extend(games)
        if meta is not None: final_meta_board = meta
        
    total_extracted_games = len(all_games)
    if total_extracted_games == 0:
        print("❌ Không thu thập được ván đấu nào! Vui lòng kiểm tra lại cấu hình đường dẫn.")
        return
        
    print(f"\n🔮 Tổng kho tích lũy: {total_extracted_games:,} ván đấu toàn cục.")
    print("🔀 Đang xáo trộn danh sách ván đấu để phá vỡ temporal leakage...")
    random.shuffle(all_games)
    
    # 3. CHIA TÁCH CẤP ĐỘ VÁN ĐẤU (90% TRAIN / 10% VAL)
    split_idx = int(0.9 * total_extracted_games)
    train_games = all_games[:split_idx]
    val_games = all_games[split_idx:]
    
    # Bung gộp mảng cho tập Train
    train_us, train_them, train_scores, train_results = [], [], [], []
    for g in train_games:
        train_us.extend(g["us_mask"])
        train_them.extend(g["them_mask"])
        train_scores.extend(g["search_score"])
        train_results.extend(g["game_result"])
        
    # Bung gộp mảng cho tập Val
    val_us_raw, val_them_raw, val_scores_raw, val_results_raw = [], [], [], []
    for g in val_games:
        val_us_raw.extend(g["us_mask"])
        val_them_raw.extend(g["them_mask"])
        val_scores_raw.extend(g["search_score"])
        val_results_raw.extend(g["game_result"])
        
    print(f"\n🛡️ Đang kích hoạt bộ lọc bảo hiểm diệt tận gốc Transposition Leakage...")
    # Tạo bảng băm nhanh từ tập Train
    train_positions_set = set(zip(train_us, train_them))
    
    val_us, val_them, val_scores, val_results = [], [], [], []
    leaked_counter = 0
    
    for u, t, s, r in zip(val_us_raw, val_them_raw, val_scores_raw, val_results_raw):
        if (u, t) in train_positions_set:
            leaked_counter += 1
        else:
            val_us.append(u)
            val_them.append(t)
            val_scores.append(s)
            val_results.append(r)
            
    print(f"   -> Đã phát hiện và gạt bỏ: {leaked_counter:,} thế cờ trùng lặp hoán vị khỏi tập Validation!")

    # 4. ĐÓNG GÓI XUẤT FILE PHÂN TÁCH SẠCH SẼ
    mask_dtype = object if final_meta_board[0] * (final_meta_board[1] + 1) > 64 else np.uint64
    
    print(f"\n💾 Đang đóng gói tệp TRAIN tổng lực (90%): '{OUTPUT_TRAIN}'...")
    np.save(OUTPUT_TRAIN, {
        "us_mask": np.array(train_us, dtype=mask_dtype),
        "them_mask": np.array(train_them, dtype=mask_dtype),
        "search_score": np.array(train_scores, dtype=np.int32),
        "game_result": np.array(train_results, dtype=np.float32),
        "meta_board": final_meta_board
    })
    
    print(f"💾 Đang đóng gói tệp VAL bảo hiểm (10%): '{OUTPUT_VAL}'...")
    np.save(OUTPUT_VAL, {
        "us_mask": np.array(val_us, dtype=mask_dtype),
        "them_mask": np.array(val_them, dtype=mask_dtype),
        "search_score": np.array(val_scores, dtype=np.int32),
        "game_result": np.array(val_results, dtype=np.float32),
        "meta_board": final_meta_board
    })
    
    print("\n=====================================================")
    print("          CHẾN DỊCH THANH LỌC DATA HOÀN TẤT")
    print("=====================================================")
    print(f"🏆 Tổng số thế cờ tập TRAIN sạch: {len(train_scores):,}")
    print(f"🏆 Tổng số thế cờ tập VAL sạch:   {len(val_scores):,}")
    print("=====================================================")

if __name__ == "__main__":
    main()