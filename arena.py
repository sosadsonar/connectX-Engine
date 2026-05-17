# arena.py
import json
import os
import time
import argparse
from board import ConnectXBoard
from ai import AdvancedNegamaxAI

def render_board(board: ConnectXBoard):
    """Giải mã Bitboard phẳng để vẽ giao diện trận đấu trực quan"""
    print("\n" + "=" * 25)
    for row in range(board.h - 1, -1, -1):
        row_str = "| "
        for col in range(board.w):
            idx = col * board.col_height + row
            mask = 1 << idx
            if board.boards[0] & mask:
                row_str += "X  "  # Bot 1 (hoặc phe đi trước)
            elif board.boards[1] & mask:
                row_str += "O  "  # Bot 2 (hoặc phe đi sau)
            else:
                row_str += ".  "  
        row_str += "|"
        print(row_str)
    print("-" * (board.w * 3 + 4))
    print("  " + "  ".join(str(i) for i in range(board.w)))
    print("=" * 25)

def load_weights(file_path):
    """Nạp bộ trọng số từ file JSON, nếu không thấy sẽ dùng cấu hình mặc định"""
    from train import BASE_CHAMPION_WEIGHTS
    if not file_path or not os.path.exists(file_path):
        print(f"[⚠️ WARNING] Không tìm thấy '{file_path}', sử dụng cấu hình BASE sơ sinh.")
        return BASE_CHAMPION_WEIGHTS
    try:
        with open(file_path, "r") as f:
            weights = json.load(f)
        print(f"[OK] Đã nạp thành công bộ não: '{file_path}'")
        return weights
    except Exception as e:
        print(f"[❌ ERROR] Lỗi đọc file {file_path}: {e}. Dùng bộ trọng số mặc định.")
        return BASE_CHAMPION_WEIGHTS

def play_single_match(w, h, x, weights_p0, weights_p1, depth0, depth1, name0, name1, time_limit, show_thinking):
    """Chạy một trận đấu đơn giữa 2 cấu hình AI"""
    board = ConnectXBoard(w, h, x)
    
    # Khởi tạo 2 AI tương ứng với 2 bộ não khác nhau
    ai0 = AdvancedNegamaxAI(weights_p0, player_id=0, tt_exponent=21)
    ai1 = AdvancedNegamaxAI(weights_p1, player_id=1, tt_exponent=21)
    
    ais = {0: ai0, 1: ai1}
    depths = {0: depth0, 1: depth1}
    names = {0: name0, 1: name1}
    symbols = {0: "X", 1: "O"}
    
    current_player = 0
    last_move = None
    move_count = 0
    
    render_board(board)
    
    while board.get_valid_cols():
        active_ai = ais[current_player]
        active_depth = depths[current_player]
        ai_name = names[current_player]
        
        print(f"\n🤖 [{ai_name}] ({symbols[current_player]}) đang tính toán...")
        start_time = time.time()
        
        # Gọi select_move với cờ verbose phụ thuộc vào cấu hình người dùng
        move = active_ai.select_move(
            board, 
            max_depth=active_depth, 
            time_limit=time_limit, 
            last_move=last_move, 
            verbose=show_thinking
        )
        
        elapsed = time.time() - start_time
        
        if move == -1:
            print(f"❌ Bot {ai_name} không thể đưa ra nước đi hợp lệ!")
            return 1 - current_player  # Đối thủ thắng
            
        board.make_move(move, current_player)
        last_move = move
        move_count += 1
        
        print(f"-> Hạ quân vào cột [{move}] (Thời gian nghĩ: {elapsed:.2f}s | Tổng số nước: {move_count})")
        render_board(board)
        
        if board.check_win(current_player):
            print(f"\n🏆 CHIẾN THẮNG THUỘC VỀ: [{ai_name}] ({symbols[current_player]}) sau {move_count} nước đi!")
            return current_player
            
        current_player = 1 - current_player
        
    print("\n🤝 Trận đấu kết thúc với kết quả HÒA (Bàn cờ đã đầy).")
    return -1

def start_arena():
    parser = argparse.ArgumentParser(description="ConnectX Hybrid Arena - Bot vs Bot Battle")
    # Tùy chỉnh kích thước địa hình bàn cờ
    parser.add_argument("--w", type=int, default=7, help="Chiều rộng bàn cờ")
    parser.add_argument("--h", type=int, default=6, help="Chiều cao bàn cờ")
    parser.add_argument("--x", type=int, default=4, help="Số quân xếp hàng để thắng")
    
    # Tùy chỉnh đường dẫn 2 file não JSON đầu vào
    parser.add_argument("--m1", type=str, default="models/best_weights_7x6_x4.json", help="Đường dẫn file JSON của Bot 1")
    parser.add_argument("--m2", type=str, default="", help="Đường dẫn file JSON của Bot 2 (để trống sẽ dùng BASE)")
    
    # Tùy chỉnh Depth tính toán riêng biệt cho từng con
    parser.add_argument("--d1", type=int, default=12, help="Depth tối đa của Bot 1")
    parser.add_argument("--d2", type=int, default=12, help="Depth tối đa của Bot 2")
    
    # Cấu hình môi trường đấu trường
    parser.add_argument("--time", type=float, default=1.8, help="Giới hạn nghĩ của Bot (giây)")
    parser.add_argument("--games", type=int, default=2, help="Số trận đấu (Nên đặt số chẵn để đổi lượt đi trước/sau cho công bằng)")
    parser.add_argument("--silent", action="store_true", help="Nếu bật, sẽ giấu tiến trình suy nghĩ chi tiết PV line của Bot, chỉ hiện kết quả hạ quân")

    args = parser.parse_args()

    print("=====================================================")
    print("      KHỞI CHẠY ĐẤU TRƯỜNG ĐỐI KHÁNG BOT VS BOT")
    print(f" Địa hình: {args.w}x{args.h} | Cấu hình Connect {args.x}")
    print("=====================================================")

    # Đọc não bộ
    weights_1 = load_weights(args.m1)
    weights_2 = load_weights(args.m2)
    
    name_1 = os.path.basename(args.m1) if args.m1 else "BASE_BOT_1"
    name_2 = os.path.basename(args.m2) if args.m2 else "BASE_BOT_2"
    
    # Thống kê điểm số tổng
    score_bot1 = 0
    score_bot2 = 0
    draws = 0
    
    show_thinking = not args.silent

    for match_idx in range(1, args.games + 1):
        print(f"\n⚔️ --- TRẬN ĐẤU THỨ {match_idx}/{args.games} --- ⚔️")
        
        # Cơ chế alternating: Đổi bên đi trước sau mỗi ván để triệt tiêu lợi thế đi trước (First-player advantage)
        if match_idx % 2 != 0:
            print(f" Phe X (Đi trước): {name_1} [Depth {args.d1}]")
            print(f" Phe O (Đi sau) : {name_2} [Depth {args.d2}]")
            result = play_single_match(
                args.w, args.h, args.x, 
                weights_1, weights_2, 
                args.d1, args.d2, 
                name_1, name_2, 
                args.time, show_thinking
            )
            if result == 0: score_bot1 += 1
            elif result == 1: score_bot2 += 1
            else: draws += 1
        else:
            print(f" Phe X (Đi trước): {name_2} [Depth {args.d2}]")
            print(f" Phe O (Đi sau) : {name_1} [Depth {args.d1}]")
            result = play_single_match(
                args.w, args.h, args.x, 
                weights_2, weights_1, 
                args.d2, args.d1, 
                name_2, name_1, 
                args.time, show_thinking
            )
            if result == 0: score_bot2 += 1
            elif result == 1: score_bot1 += 1
            else: draws += 1

    print("\n=====================================================")
    print("               BẢNG TỔNG KẾT CHUNG CUỘC")
    print("=====================================================")
    print(f"🏆 [{name_1}] (Depth {args.d1}): {score_bot1} Trận thắng")
    print(f"🏆 [{name_2}] (Depth {args.d2}): {score_bot2} Trận thắng")
    print(f"🤝 Số trận HÒA: {draws}")
    print("=====================================================")

if __name__ == "__main__":
    start_arena()
