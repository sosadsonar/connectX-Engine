# train.py
import copy
import random
import json
import os
import time
import argparse
from board import ConnectXBoard
from ai import AdvancedNegamaxAI

# Sửa lại bộ trọng số gốc trong train.py để Bot nhạy cảm hơn với thế trận của bạn
BASE_CHAMPION_WEIGHTS = {
    "WIN_BASE": 10000000,
    "FORK_SCORE": 600000,      # Tăng điểm thưởng khi nhìn ra thế đe dọa kép
    "THREAT_SCORE": 25000,    
    "MAX_STRATEGIC": 15000,   
    "C_SMOOTH": 2000.0,       
    "K_EDGE": 30.0,            # Tăng điểm ô cạnh để ưu tiên bám đuổi hàng ngang
    "K_CORNER": 10.0,         
    "CNN_POWER": 2.2,          # Tăng lũy thừa để Bot thích tụ quân bầy đàn
    "ALPHA_BALANCED": 1.0,    
    "ALPHA_DEFENSIVE": 1.8,    # Tăng mạnh hệ số sợ hãi khi bị người chơi ép sân
    "ASPIRATION_DELTA": 3000.0  
}

def mutate_weights(base_weights, annealing_factor):
    mutated = copy.deepcopy(base_weights)
    target_keys = ["FORK_SCORE", "THREAT_SCORE", "MAX_STRATEGIC", "C_SMOOTH", "K_EDGE", "K_CORNER", "CNN_POWER", "ALPHA_BALANCED", "ALPHA_DEFENSIVE", "ASPIRATION_DELTA"]
    for key in target_keys:
        factor = random.uniform(1 - annealing_factor, 1 + annealing_factor)
        mutated[key] = type(mutated[key])(mutated[key] * factor)
    return mutated

def run_match_arena(w, h, x, weights_p0, weights_p1, max_depth) -> int:
    board = ConnectXBoard(w, h, x)
    ai0 = AdvancedNegamaxAI(weights_p0, player_id=0, tt_exponent=21)
    ai1 = AdvancedNegamaxAI(weights_p1, player_id=1, tt_exponent=21)
    
    current_player = 0
    ais = {0: ai0, 1: ai1}
    last_move = None
    
    while board.get_valid_cols():
        # KHÓA CHÍ MẠNG: verbose=False giúp tắt toàn bộ log rác trong lúc train
        move = ais[current_player].select_move(board, max_depth=max_depth, time_limit=1.8, last_move=last_move, verbose=False)
        if move == -1:
            break
            
        board.make_move(move, current_player)
        last_move = move
        if board.check_win(current_player):
            return current_player
            
        current_player = 1 - current_player
        
    return -1

def save_optimized_weights(champion_weights, w, h, x, search_depth, model_dir="models", custom_filename=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    file_name = custom_filename if custom_filename else f"best_weights_{w}x{h}_x{x}.json"
    file_path = os.path.join(model_dir, file_name)
    
    data_to_save = copy.deepcopy(champion_weights)
    data_to_save["meta_target_board"] = f"{w}x{h}"
    data_to_save["meta_target_x"] = x
    data_to_save["meta_trained_depth"] = search_depth

    should_overwrite = True
    if not custom_filename and os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                existing_data = json.load(f)
            old_depth = existing_data.get("meta_trained_depth", 0)
            if old_depth > search_depth:
                should_overwrite = False
        except Exception:
            pass

    if should_overwrite:
        with open(file_path, "w") as f:
            json.dump(data_to_save, f, indent=4)
        if not custom_filename:
            print(f" -> [LƯU TRỮ] Đã cập nhật cấu hình vương quyền mới vĩnh viễn!")

def start_hardcore_training(generations, w, h, x, search_depth, cp_interval, model_dir="models"):
    champion = BASE_CHAMPION_WEIGHTS
    
    file_path = os.path.join(model_dir, f"best_weights_{w}x{h}_x{x}.json")
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                loaded_data = json.load(f)
            for key in champion.keys():
                if key in loaded_data:
                    champion[key] = loaded_data[key]
            print(f"[HỆ THỐNG] Nạp bộ trọng số tối ưu cũ làm bệ phóng.\n")
        except Exception:
            pass

    print(f"====================================================")
    print(f" KHỞI CHẠY TIẾN HÓA SẠCH (PVS + IDS + ASPIRATION)")
    print(f" Địa hình: {w}x{h} | Connect {x} | Giới hạn Depth: {search_depth}")
    print(f" CHẾ ĐỘ TERMINAL: IM LẶNG TUYỆT ĐỐI (SILENT ENGINE)")
    print(f"====================================================\n")
    
    start_time = time.time()

    for gen in range(1, generations + 1):
        annealing_factor = max(0.01, 0.08 * (1 - (gen / generations)))
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
            print(f"[Thế hệ {gen:03d}/{generations}] -> 🔥 ĐỔI NGÔI VƯƠNG! Tỷ số: {chal_score}-{champ_score} | Aspiration Delta: {champion['ASPIRATION_DELTA']:.1f}")
            save_optimized_weights(champion, w, h, x, search_depth, model_dir)
        else:
            if gen % 10 == 0 or gen == 1:
                elapsed = time.time() - start_time
                print(f"[Tiến độ {gen:03d}/{generations}] Nhà vô địch thủ ngôi ổn định. Tốc độ hiển thị: {gen / elapsed:.2f} gen/giây.")

        if gen % cp_interval == 0:
            cp_filename = f"checkpoint_{w}x{h}_x{x}_gen_{gen}.json"
            save_optimized_weights(champion, w, h, x, search_depth, model_dir, custom_filename=cp_filename)

    save_optimized_weights(champion, w, h, x, search_depth, model_dir)
    print(f"\n Toàn bộ chiến dịch hoàn tất sạch sẽ trong {(time.time() - start_time)/60:.2f} phút.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ConnectX Silent Engine Training System")
    parser.add_argument("--w", type=int, default=7)
    parser.add_argument("--h", type=int, default=6)
    parser.add_argument("--x", type=int, default=4)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--gen", type=int, default=100)
    parser.add_argument("--cp", type=int, default=20)
    
    args = parser.parse_args()
    start_hardcore_training(generations=args.gen, w=args.w, h=args.h, x=args.x, search_depth=args.depth, cp_interval=args.cp)