# train.py
import copy
import random
import json
import os
import time
import argparse
from board import ConnectXBoard
from ai import AdvancedNegamaxAI

# Bộ trọng số tinh anh làm gốc để liên tục tiến hóa
BASE_CHAMPION_WEIGHTS = {
    "WIN_BASE": 10000000,
    "FORK_SCORE": 848725,
    "THREAT_SCORE": 53116,
    "MAX_STRATEGIC": 47279,
    "C_SMOOTH": 4214.353518903514,
    "K_EDGE": 102.48017570159823,
    "K_CORNER": 59.342997052116836,
    "CNN_POWER": 1.8029926717664617,
    "ALPHA_BALANCED": 0.9163077283542962,
    "ALPHA_DEFENSIVE": 1.385254389243149
}

def mutate_weights(base_weights, annealing_factor):
    """Vi chỉnh trọng số siêu nhỏ để dò tìm siêu tối ưu"""
    mutated = copy.deepcopy(base_weights)
    target_keys = ["FORK_SCORE", "THREAT_SCORE", "MAX_STRATEGIC", "C_SMOOTH", "K_EDGE", "K_CORNER", "CNN_POWER", "ALPHA_BALANCED", "ALPHA_DEFENSIVE"]
    for key in target_keys:
        factor = random.uniform(1 - annealing_factor, 1 + annealing_factor)
        mutated[key] = type(mutated[key])(mutated[key] * factor)
    return mutated

def run_match_arena(w, h, x, weights_p0, weights_p1, max_depth) -> int:
    """Điều phối trận đấu đối kháng Bitboard"""
    board = ConnectXBoard(w, h, x)
    ai0 = AdvancedNegamaxAI(weights_p0, player_id=0, tt_exponent=21)
    ai1 = AdvancedNegamaxAI(weights_p1, player_id=1, tt_exponent=21)
    
    current_player = 0
    ais = {0: ai0, 1: ai1}
    
    while board.get_valid_cols():
        move = ais[current_player].select_move(board, max_depth=max_depth, time_limit=1.8)
        if move == -1:
            break
            
        board.make_move(move, current_player)
        if board.check_win(current_player):
            return current_player
            
        current_player = 1 - current_player
        
    return -1

def save_optimized_weights(champion_weights, w, h, x, search_depth, model_dir="models", custom_filename=None):
    """Hàm quản lý lưu trữ: Tách biệt kích thước, checkpoint và ưu tiên Depth cao đè depth thấp"""
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    # Phân loại: Nếu có tên custom (Checkpoint) thì dùng luôn, nếu không thì dùng file "best" mặc định
    file_name = custom_filename if custom_filename else f"best_weights_{w}x{h}_x{x}.json"
    file_path = os.path.join(model_dir, file_name)
    
    data_to_save = copy.deepcopy(champion_weights)
    data_to_save["meta_target_board"] = f"{w}x{h}"
    data_to_save["meta_target_x"] = x
    data_to_save["meta_trained_depth"] = search_depth

    should_overwrite = True
    
    # CHỈ áp dụng luật so sánh depth chặn đè cho file "best_weights" tổng hợp
    if not custom_filename and os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                existing_data = json.load(f)
            old_depth = existing_data.get("meta_trained_depth", 0)
            if old_depth > search_depth:
                should_overwrite = False
                print(f" -> [LƯU TRỮ] Giữ file cũ vì độ sâu cũ ({old_depth}) cao hơn độ sâu hiện tại ({search_depth}).")
        except Exception:
            pass

    if should_overwrite:
        with open(file_path, "w") as f:
            json.dump(data_to_save, f, indent=4)
        if not custom_filename:
            print(f" -> [LƯU TRỮ] Đã cập nhật cấu hình tối ưu vĩnh viễn: '{file_path}' (Depth: {search_depth})")
        else:
            print(f" --> [SYSTEM] Đã lưu checkpoint an toàn: '{file_path}'")

def start_hardcore_training(generations, w, h, x, search_depth, cp_interval, model_dir="models"):
    champion = BASE_CHAMPION_WEIGHTS
    
    # Nạp lại trọng số nền cũ nếu có để tiếp tục tiến hóa gối đầu
    file_path = os.path.join(model_dir, f"best_weights_{w}x{h}_x{x}.json")
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                loaded_data = json.load(f)
            for key in champion.keys():
                if key in loaded_data:
                    champion[key] = loaded_data[key]
            print(f"[HỆ THỐNG] Nạp thành công bộ trọng số nền đã tối ưu của bàn cờ {w}x{h}_x{x}.\n")
        except Exception:
            pass

    print(f"====================================================")
    print(f" KÍCH HOẠT TIẾN HÓA ĐỘNG LỰC HỌC: {generations} THẾ HỆ")
    print(f" Địa hình: {w}x{h} | Mục tiêu: Connect {x}")
    print(f" ĐỘ SÂU GIỚI HẠN (MAX DEPTH): TẦNG {search_depth}")
    print(f" CHECKPOINT INTERVAL: TỰ ĐỘNG SAU MỖI {cp_interval} THẾ HỆ")
    print(f"====================================================\n")
    
    start_time = time.time()

    for gen in range(1, generations + 1):
        annealing_factor = max(0.005, 0.03 * (1 - (gen / generations)))
        challenger = mutate_weights(champion, annealing_factor)
        
        champ_score = 0
        chal_score = 0
        
        res1 = run_match_arena(w, h, x, champion, challenger, max_depth=search_depth)
        if res1 == 0: champ_score += 1
        elif res1 == 1: chal_score += 1
        
        res2 = run_match_arena(w, h, x, challenger, champion, max_depth=search_depth)
        if res2 == 0: chal_score += 1
        elif res2 == 1: champ_score += 1

        if chal_score > champ_score:
            champion = challenger
            print(f"[Gen {gen:03d}/{generations}] -> THÁCH THỨC CHIẾM NGÔI! Tỷ số: {chal_score}-{champ_score}")
            save_optimized_weights(champion, w, h, x, search_depth, model_dir)
        else:
            if gen % 5 == 0 or gen == 1:
                elapsed = time.time() - start_time
                print(f"[Thế hệ {gen:03d}/{generations}] Trận đấu kết thúc ổn định. Vận tốc: {gen / elapsed:.2f} gen/giây.")

        # TỰ ĐỘNG TRÍCH XUẤT CHECKPOINT THEO THAM SỐ TÙY CHỈNH
        if gen % cp_interval == 0:
            cp_filename = f"checkpoint_{w}x{h}_x{x}_gen_{gen}.json"
            save_optimized_weights(champion, w, h, x, search_depth, model_dir, custom_filename=cp_filename)

    # Lưu phát cuối khi kết thúc chiến dịch
    save_optimized_weights(champion, w, h, x, search_depth, model_dir)
    print(f"\n Chiến dịch hoàn tất hoàn hảo trong { (time.time() - start_time)/60:.2f} phút.")

if __name__ == "__main__":
    # Cấu trúc nhận diện toàn diện tham số động (Đã loại bỏ dòng lỗi cũ)
    parser = argparse.ArgumentParser(description="ConnectX Bitboard CNN Tuning Engine")
    parser.add_argument("--w", type=int, default=7, help="Chiều rộng bàn cờ")
    parser.add_argument("--h", type=int, default=6, help="Chiều cao bàn cờ")
    parser.add_argument("--x", type=int, default=4, help="Số quân để thắng")
    parser.add_argument("--depth", type=int, default=12, help="Độ sâu tìm kiếm tối đa (IDS tự động)")
    parser.add_argument("--gen", type=int, default=100, help="Số thế hệ huấn luyện")
    parser.add_argument("--cp", type=int, default=20, help="Khoảng cách thế hệ để lưu một file Checkpoint")
    
    args = parser.parse_args()
    
    start_hardcore_training(
        generations=args.gen, 
        w=args.w, 
        h=args.h, 
        x=args.x, 
        search_depth=args.depth, 
        cp_interval=args.cp
    )