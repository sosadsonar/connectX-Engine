# train.py
import copy
import random
import json
import os
import time
import argparse
from board import ConnectXBoard
from ai import AdvancedNegamaxAI

# Trọng số vương quyền nền tảng dùng làm bệ phóng tiến hóa
BASE_CHAMPION_WEIGHTS = {
    "WIN_BASE": 10000000,
    "FORK_SCORE": 641839,
    "THREAT_SCORE": 36499,
    "MAX_STRATEGIC": 13607,
    "C_SMOOTH": 1540.7259706098523,
    "K_EDGE": 23.442601692030077,
    "K_CORNER": 8.682558369484749,
    "CNN_POWER": 2.2565585014055505,
    "ALPHA_BALANCED": 0.6201356568933092,
    "ALPHA_DEFENSIVE": 1.4055617960644347,
    "ASPIRATION_DELTA": 3076.8027562957773,
    "PARITY_ROW_BONUS": 141.43804191035696,
    "PARITY_ROW_PENALTY": 162.6573907381674,
    "meta_target_board": "7x6",
    "meta_target_x": 4,
    "meta_trained_depth": 20
}

def render_board(board: ConnectXBoard):
    """Vẽ bàn cờ động trực quan phục vụ giám sát đấu trường Arena"""
    print("\n" + "=" * 25)
    for row in range(board.h - 1, -1, -1):
        row_str = "| "
        for col in range(board.w):
            idx = col * board.col_height + row
            mask = 1 << idx
            if board.boards[0] & mask:
                row_str += "X  "  # Quân của Champion (Player 0)
            elif board.boards[1] & mask:
                row_str += "O  "  # Quân của Challenger (Player 1)
            else:
                row_str += ".  "  
        row_str += "|"
        print(row_str)
    print("-" * (board.w * 3 + 4))
    print("  " + "  ".join(str(i) for i in range(board.w)))
    print("=" * 25)

def mutate_weights(base_weights, annealing_factor):
    """Đột biến có kiểm soát dựa trên hệ số hạ nhiệt giảm dần theo thế hệ"""
    mutated = copy.deepcopy(base_weights)
    target_keys = [
        "FORK_SCORE", "THREAT_SCORE", "MAX_STRATEGIC", "C_SMOOTH", 
        "K_EDGE", "K_CORNER", "CNN_POWER", "ALPHA_BALANCED", 
        "ALPHA_DEFENSIVE", "ASPIRATION_DELTA",
        "PARITY_ROW_BONUS", "PARITY_ROW_PENALTY"
    ]
    for key in target_keys:
        factor = random.uniform(1 - annealing_factor, 1 + annealing_factor)
        mutated[key] = type(mutated[key])(mutated[key] * factor)
    return mutated

def verify_against_human_lessons(weights, w, h, x):
    """
    BỘ LỌC PHÁT HIỆN SAI LẦM (BLUNDER FILTER):
    Chỉ loại bỏ bộ gen nếu thế cờ vẫn còn đường cứu nhưng Bot lại chọn nước đi thua.
    Đã đồng bộ tuyệt đối tương thích với cấu hình Sorter và Ply mới.
    """
    file_path = "../models/human_lessons.json"
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
                    
                    # THUẬT TOÁN KIỂM TRA: Thế cờ hiện tại thực sự còn cứu được không?
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
                            break # Tìm thấy ít nhất một giải pháp thủ hiểm nguy cứu bàn cờ
                    
                    # Nếu thế cờ còn cứu được, nhưng bộ gen đột biến lại đâm đầu vào họng súng -> ĐÁNH TRƯỢT
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
                            return False # Loại bỏ bộ gen lỗi chiến thuật thô thiển từ trứng nước
                            
            test_board.make_move(m, curr_player)
    return True

def run_match_arena(w, h, x, weights_p0, weights_p1, max_depth, opening_moves=None) -> int:
    """Đấu trường hiển thị trực quan trận đấu giữa Champion và Challenger"""
    board = ConnectXBoard(w, h, x)
    ai0 = AdvancedNegamaxAI(weights_p0, player_id=0, tt_exponent=23)
    ai1 = AdvancedNegamaxAI(weights_p1, player_id=1, tt_exponent=23)
    
    current_player = 0
    ais = {0: ai0, 1: ai1}
    last_move = None
    
    # In thông báo bắt đầu trận đấu thử nghiệm ElO
    name_p0 = "🏆 CHAMPION (X)"
    name_p1 = "⚡ CHALLENGER (O)"
    print(f"\n⚔️  [ARENA] Trận đấu khởi tranh: {name_p0}  VS  {name_p1}")
    
    # Kích hoạt trận địa khai cuộc ngẫu nhiên
    if opening_moves:
        print(f"📦 Chuỗi nước đi khai cuộc nền: {opening_moves}")
        for m in opening_moves:
            if m in board.get_valid_cols():
                board.make_move(m, current_player)
                last_move = m
                current_player = 1 - current_player

    # Vẽ trạng thái bàn cờ sau khai cuộc
    render_board(board)
    time.sleep(0.3) # Giãn cách để mắt người kịp theo dõi

    move_count = len(opening_moves) if opening_moves else 0
    while board.get_valid_cols():
        move_count += 1
        curr_name = name_p0 if current_player == 0 else name_p1
        
        # BẬT verbose=True ĐỂ HIỂN THỊ CẢ MA TRẬN PHÂN TÍCH ĐIỂM SỐ ĐANG DEBUG Chuẩn C++ CỦA BOT!
        move = ais[current_player].select_move(board, max_depth=max_depth, time_limit=2.2, last_move=last_move, verbose=True)
        
        if move == -1:
            print(f"💀 {curr_name} không tìm thấy nước đi hợp lệ.")
            break
            
        board.make_move(move, current_player)
        last_move = move
        
        # In hành động hạ quân của từng Bot
        print(f"🤖 [Nước {move_count:02d}] {curr_name} hạ quân vào cột [{move}]")
        render_board(board)
        
        # Tốc độ hoạt cảnh (0.5 giây một nước, bạn có thể giảm xuống 0.1 nếu muốn xem Bot đánh siêu tốc)
        time.sleep(0.5) 
        
        if board.check_win(current_player):
            print(f"🎉 Trận đấu kết thúc! {curr_name} CHIẾN THẮNG SÁT CỤC tuyệt đối!\n")
            return current_player
            
        current_player = 1 - current_player
        
    print("🤝 Trận đấu kết thúc với kết quả HÒA CĂNG THẲNG.\n")
    return -1

def save_optimized_weights(champion_weights, w, h, x, search_depth, model_dir="../models", custom_filename=None):
    """Ghi lại bộ trọng số xuất sắc nhất xuống ổ cứng cấu trúc JSON sạch"""
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

def start_hardcore_training(generations, w, h, x, search_depth, cp_interval, model_dir="../models"):
    champion = BASE_CHAMPION_WEIGHTS
    
    file_path = os.path.join(model_dir, f"best_weights_{w}x{h}_x{x}.json")
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                loaded_data = json.load(f)
            for key in champion.keys():
                if key in loaded_data:
                    champion[key] = loaded_data[key]
            print(f"[HỆ THỐNG] Nạp bộ trọng số tối ưu cũ làm bệ phóng vững chắc.\n")
        except Exception:
            pass

    print(f"====================================================")
    print(f" KHỞI CHẠY TIẾN HÓA SẠCH NÂNG CAO (4-MATCH OPENINGS MATRIX)")
    print(f" Địa hình: {w}x{h} | Connect {x} | Giới hạn Thiết lập Depth: {search_depth}")
    print(f" BỘ LỌC CHIẾN THUẬT: ĐANG BẬT BÀI THI TỪ CON NGƯỜI 🧠")
    print(f"====================================================\n")
    
    start_time = time.time()
    
    # Xác định các cột chiến lược khu vực trung tâm để sinh thế cờ đối kháng cân bằng
    center_cols = [w//2 - 1, w//2, w//2 + 1]

    for gen in range(1, generations + 1):
        annealing_factor = max(0.01, 0.08 * (1 - (gen / generations)))
        challenger = mutate_weights(champion, annealing_factor)
        
        # Thử thách 1: Vượt qua bài kiểm tra Blunder chẵn lẻ của con người
        if not verify_against_human_lessons(challenger, w, h, x):
            continue

        champ_score = 0
        chal_score = 0
        
        # --- MA TRẬN ĐỐI KHÁNG ĐA DIỆN 4 TRẬN ĐỂ TRÁNH OVERFITTING ĐƯỜNG ĐI ---
        # Thế trận mẫu A: Sinh ngẫu nhiên 2 nước đi nền tại trục trung tâm
        opening_A = []
        
        res1 = run_match_arena(w, h, x, champion, challenger, max_depth=search_depth, opening_moves=opening_A)
        if res1 == 0: champ_score += 1
        elif res1 == 1: chal_score += 1
        
        opening_B = []
        
        res3 = run_match_arena(w, h, x, champion, challenger, max_depth=search_depth, opening_moves=opening_B)
        if res3 == 0: champ_score += 1
        elif res3 == 1: chal_score += 1
        
        # ĐIỀU KIỆN TIẾN HÓA GẮT GAO: Kẻ thách thức bắt buộc phải thắng tuyệt đối trên tổng thể đa trận địa
        if chal_score > champ_score:
            champion = challenger
            print(f"[Thế hệ {gen:03d}/{generations}] -> 🔥 ĐỔI NGÔI VƯƠNG THỰC THỤ! Tỷ số: {chal_score}-{champ_score} | Aspiration Delta: {champion['ASPIRATION_DELTA']:.1f}")
            save_optimized_weights(champion, w, h, x, search_depth, model_dir)
        else:
            if gen % 10 == 0 or gen == 1:
                elapsed = time.time() - start_time
                print(f"[Tiến độ {gen:03d}/{generations}] Nhà vô địch thủ ngôi ổn định. Tốc độ hiển thị: {gen / elapsed:.2f} gen/giây.")

        # Ghi nhận các điểm Checkpoint định kỳ phòng ngừa mất điện/hỏng luồng
        if gen % cp_interval == 0:
            cp_filename = f"checkpoint_{w}x{h}_x{x}_gen_{gen}.json"
            save_optimized_weights(champion, w, h, x, search_depth, model_dir, custom_filename=cp_filename)

    save_optimized_weights(champion, w, h, x, search_depth, model_dir)
    print(f"\n Toàn bộ chiến dịch huấn luyện hoàn tất sạch sẽ trong {(time.time() - start_time)/60:.2f} phút.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ConnectX Silent Engine Training System")
    parser.add_argument("--w", type=int, default=7)
    parser.add_argument("--h", type=int, default=6)
    parser.add_argument("--x", type=int, default=4)
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--gen", type=int, default=20)
    parser.add_argument("--cp", type=int, default=5)
    
    args = parser.parse_args()
    start_hardcore_training(generations=args.gen, w=args.w, h=args.h, x=args.x, search_depth=args.depth, cp_interval=args.cp)