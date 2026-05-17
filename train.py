# train.py
import copy
import random
import json
import os
import time
import argparse
from board import ConnectXBoard
from ai import AdvancedNegamaxAI

BASE_CHAMPION_WEIGHTS = {
    "WIN_BASE": 10000000,
    "FORK_SCORE": 696285,
    "THREAT_SCORE": 23562,
    "MAX_STRATEGIC": 13399,
    "C_SMOOTH": 1761.419889601332,
    "K_EDGE": 23.78560574060163,
    "K_CORNER": 9.751297449273448,
    "CNN_POWER": 2.513904856699739,
    "ALPHA_BALANCED": 1.005723998128732,
    "ALPHA_DEFENSIVE": 1.8969864687039997,
    "ASPIRATION_DELTA": 2759.0853022807087
}

def mutate_weights(base_weights, annealing_factor):
    mutated = copy.deepcopy(base_weights)
    target_keys = ["FORK_SCORE", "THREAT_SCORE", "MAX_STRATEGIC", "C_SMOOTH", "K_EDGE", "K_CORNER", "CNN_POWER", "ALPHA_BALANCED", "ALPHA_DEFENSIVE", "ASPIRATION_DELTA"]
    for key in target_keys:
        factor = random.uniform(1 - annealing_factor, 1 + annealing_factor)
        mutated[key] = type(mutated[key])(mutated[key] * factor)
    return mutated

def verify_against_human_lessons(weights, w, h, x):
    """
    BỘ LỌC PHÁT HIỆN SAI LẦM (BLUNDER FILTER):
    Chỉ loại bỏ bộ gen nếu thế cờ vẫn còn đường cứu chịu nhưng Bot lại chọn nước đi thua.
    Nếu thế cờ là sát cục ép buộc (Forced Loss), cho phép đi tiếp.
    """
    file_path = "models/human_lessons.json"
    if not os.path.exists(file_path):
        return True

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lessons = json.load(f)
    except:
        return True

    tester = AdvancedNegamaxAI(weights, player_id=0, tt_exponent=15)

    for lesson in lessons:
        if lesson["w"] != w or lesson["h"] != h or lesson["x"] != x:
            continue
            
        test_board = ConnectXBoard(w, h, x)
        moves = lesson["moves"]
        human_id = lesson["human_id"]
        bot_id = 1 - human_id
        
        for turn_idx, m in enumerate(moves):
            if not test_board.get_valid_cols():
                break
            curr_player = 0 if turn_idx % 2 == 0 else 1
            
            # Quét các nước đi ở giai đoạn tàn cuộc quyết định của trận đấu cũ
            if curr_player == bot_id and turn_idx >= len(moves) - 6:
                tester.player_id = bot_id
                chosen_move = tester.select_move(test_board, max_depth=4, time_limit=0.2, verbose=False)
                
                if chosen_move != -1 and chosen_move in test_board.get_valid_cols():
                    
                    # THUẬT TOÁN KIỂM TRA: Thế cờ hiện tại còn cứu được không?
                    any_move_saves = False
                    for c_bot in test_board.get_valid_cols():
                        test_board.make_move(c_bot, bot_id)
                        human_can_win_immediately = False
                        for c_hum in test_board.get_valid_cols():
                            test_board.make_move(c_hum, human_id)
                            if test_board.check_win(human_id):
                                human_can_win_immediately = True
                            test_board.undo_move(c_hum, human_id)
                        test_board.undo_move(c_bot, bot_id)
                        
                        if not human_can_win_immediately:
                            any_move_saves = True
                            break # Tìm thấy ít nhất một nước đi giúp Bot thủ được qua lượt này
                    
                    # Nếu thế cờ còn cứu được (any_move_saves == True), 
                    # nhưng nước đi thực tế của Bộ gene đột biến này lại để lộ họng súng -> ĐÁNH TRƯỢT
                    if any_move_saves:
                        test_board.make_move(chosen_move, bot_id)
                        blunder = False
                        for next_col in test_board.get_valid_cols():
                            test_board.make_move(next_col, human_id)
                            if test_board.check_win(human_id):
                                blunder = True
                            test_board.undo_move(next_col, human_id)
                        test_board.undo_move(chosen_move, bot_id)
                        
                        if blunder:
                            return False # Loại bỏ bộ gen lỗi chiến thuật thô thiển
                            
            test_board.make_move(m, curr_player)
    return True

def run_match_arena(w, h, x, weights_p0, weights_p1, max_depth) -> int:
    board = ConnectXBoard(w, h, x)
    ai0 = AdvancedNegamaxAI(weights_p0, player_id=0, tt_exponent=21)
    ai1 = AdvancedNegamaxAI(weights_p1, player_id=1, tt_exponent=21)
    
    current_player = 0
    ais = {0: ai0, 1: ai1}
    last_move = None
    
    while board.get_valid_cols():
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
    print(f" BỘ LỌC KIỂM TRA: ĐANG BẬT BÀI THI TỪ CON NGƯỜI 🧠")
    print(f"====================================================\n")
    
    start_time = time.time()

    for gen in range(1, generations + 1):
        annealing_factor = max(0.01, 0.08 * (1 - (gen / generations)))
        challenger = mutate_weights(champion, annealing_factor)
        
        # Gọi bộ lọc thông minh đã được nâng cấp sửa lỗi
        if not verify_against_human_lessons(challenger, w, h, x):
            continue

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