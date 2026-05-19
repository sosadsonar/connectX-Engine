# arena.py
import json
import os
import time
import random
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
                row_str += "X  "  
            elif board.boards[1] & mask:
                row_str += "O  "  
            else:
                row_str += ".  "  
        row_str += "|"
        print(row_str)
    print("-" * (board.w * 3 + 4))
    print("  " + "  ".join(str(i) for i in range(board.w)))
    print("=" * 25)

def load_weights(file_path):
    """Nạp bộ trọng số từ file JSON cho chế độ Heuristic và Hybrid"""
    if not file_path or not os.path.exists(file_path):
        from train import BASE_CHAMPION_WEIGHTS
        return BASE_CHAMPION_WEIGHTS
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        from train import BASE_CHAMPION_WEIGHTS
        print(f"[❌ ERROR] Lỗi đọc file JSON {file_path}: {e}")
        return BASE_CHAMPION_WEIGHTS

def play_single_match(w, h, x, w_p0, w_p1, mode0, mode1, depth0, depth1, name0, name1, time_limit, show_thinking, opening_moves):
    """Chạy một trận đấu đơn giữa 2 cấu hình AI độc lập"""
    board = ConnectXBoard(w, h, x)
    
    # Khởi tạo 2 AI tương thích đa nền tảng (Heuristic / NNUE / Hybrid)
    ai0 = AdvancedNegamaxAI(w_p0, player_id=0, mode=mode0, tt_exponent=23)
    ai1 = AdvancedNegamaxAI(w_p1, player_id=1, mode=mode1, tt_exponent=23)
    
    ais = {0: ai0, 1: ai1}
    depths = {0: depth0, 1: depth1}
    names = {0: name0, 1: name1}
    symbols = {0: "X", 1: "O"}
    
    current_player = 0
    last_move = None
    move_count = 0
    
    render_board(board)
    
    while board.get_valid_cols():
        valid_cols = board.get_valid_cols()
        active_ai = ais[current_player]
        active_depth = depths[current_player]
        ai_name = names[current_player]
        
        print(f"\n🤖 [{ai_name}] ({symbols[current_player]}) đang tính toán...")
        start_time = time.time()
        
        # 🎯 CƠ CHẾ PHÁ BẪY TRÙNG LẶP KHAI CUỘC
        if move_count < opening_moves:
            move = random.choice(valid_cols)
            print(f"🎲 [🎲 KHAI CUỘC NGẪU NHIÊN] Thả tự do quân cờ vào cột [{move}]")
        else:
            move = active_ai.select_move(
                board, 
                max_depth=active_depth, 
                time_limit=time_limit, 
                last_move=last_move, 
                verbose=show_thinking
            )
        
        elapsed = time.time() - start_time
        
        if move == -1:
            print(f"❌ Bot {ai_name} ngáo ngơ dính lỗi không thể đưa ra nước đi!")
            return 1 - current_player  
            
        board.make_move(move, current_player)
        last_move = move
        move_count += 1
        
        print(f"-> Hạ quân vào cột [{move}] (Thời gian: {elapsed:.2f}s | Tổng số nước: {move_count})")
        render_board(board)
        
        if board.check_win(current_player):
            print(f"\n🏆 CHIẾN THẮNG THUỘC VỀ: [{ai_name}] ({symbols[current_player]}) sau {move_count} nước!")
            return current_player
            
        current_player = 1 - current_player
        
    print("\n🤝 Trận đấu kết thúc với kết quả HÒA (Bàn cờ đã đầy).")
    return -1

def start_arena():
    parser = argparse.ArgumentParser(description="ConnectX Production Arena v2.2 - Gated Hybrid Battle")
    parser.add_argument("--w", type=int, default=7)
    parser.add_argument("--h", type=int, default=6)
    parser.add_argument("--x", type=int, default=4)
    
    # ⚙️ MỞ RỘNG CỔNG NHẬN CHẾ ĐỘ: Thêm "hybrid" vào danh sách choices uy tín
    parser.add_argument("--mode1", type=str, default="hybrid", choices=["heuristic", "nnue", "hybrid"], help="Chế độ chạy của Bot 1")
    parser.add_argument("--mode2", type=str, default="heuristic", choices=["heuristic", "nnue", "hybrid"], help="Chế độ chạy của Bot 2")
    
    parser.add_argument("--m1", type=str, default="../models/best_weights_7x6_x4.json", help="File JSON của Bot 1 (Nếu chạy Heuristic/Hybrid)")
    parser.add_argument("--m2", type=str, default="../models/best_weights_7x6_x4.json", help="File JSON của Bot 2 (Nếu chạy Heuristic/Hybrid)")
    
    parser.add_argument("--d1", type=int, default=12, help="Depth tối đa của Bot 1")
    parser.add_argument("--d2", type=int, default=12, help="Depth tối đa của Bot 2")
    
    parser.add_argument("--time", type=float, default=1.8)
    parser.add_argument("--games", type=int, default=10, help="Nên đặt số chẵn để luân phiên đổi vế đi trước/sau")
    parser.add_argument("--opening_moves", type=int, default=4, help="Số nước đi ngẫu nhiên lúc khai cuộc")
    parser.add_argument("--silent", action="store_true")

    args = parser.parse_args()

    print("=====================================================")
    print("      🔥 ĐẦU TRƯỜNG KHAI HỎA: ĐỐI KHÁNG LAI MÃ v2.2")
    print(f" Thử nghiệm: Bot 1 [{args.mode1.upper()}] (D{args.d1}) ⚔️ Bot 2 [{args.mode2.upper()}] (D{args.d2})")
    print(f" Cấu hình bọc lót: Lọc bẫy Fork từ ngưỡng >= 500k điểm 🛡️")
    print("=====================================================\n")

    # Nạp tài nguyên thô (Chế độ Hybrid bắt buộc phải có JSON của HCE đi kèm để chạy bộ lọc cổng gác)
    w_p0 = load_weights(args.m1) if args.mode1 in ["heuristic", "hybrid"] else None
    w_p1 = load_weights(args.m2) if args.mode2 in ["heuristic", "hybrid"] else None
    
    name_1 = f"Bot_HYBRID_V2" if args.mode1 == "hybrid" else (f"Bot_NNUE_V2" if args.mode1 == "nnue" else os.path.basename(args.m1))
    name_2 = f"Bot_HYBRID_V2" if args.mode2 == "hybrid" else (f"Bot_NNUE_V2" if args.mode2 == "nnue" else os.path.basename(args.m2))
    
    score_bot1 = 0
    score_bot2 = 0
    draws = 0
    show_thinking = not args.silent

    for match_idx in range(1, args.games + 1):
        print(f"\n⚔️ --- TRẬN ĐẤU THỨ {match_idx}/{args.games} --- ⚔️")
        
        # Luân phiên đổi bên đi trước sau mỗi ván để đảm bảo tính khách quan
        if match_idx % 2 != 0:
            print(f" Phe X (Đi trước): {name_1} [{args.mode1.upper()}]")
            print(f" Phe O (Đi sau) : {name_2} [{args.mode2.upper()}]")
            result = play_single_match(
                args.w, args.h, args.x, w_p0, w_p1, 
                args.mode1, args.mode2, args.d1, args.d2, 
                name_1, name_2, args.time, show_thinking, args.opening_moves
            )
            if result == 0: score_bot1 += 1
            elif result == 1: score_bot2 += 1
            else: draws += 1
        else:
            print(f" Phe X (Đi trước): {name_2} [{args.mode2.upper()}]")
            print(f" Phe O (Đi sau) : {name_1} [{args.mode1.upper()}]")
            result = play_single_match(
                args.w, args.h, args.x, w_p1, w_p0, 
                args.mode2, args.mode1, args.d2, args.d1, 
                name_2, name_1, args.time, show_thinking, args.opening_moves
            )
            if result == 0: score_bot2 += 1
            elif result == 1: score_bot1 += 1
            else: draws += 1

    print("\n=====================================================")
    print("               BẢNG TỔNG KẾT CHUNG CUỘC")
    print("=====================================================")
    print(f"🏆 [{name_1}] ({args.mode1.upper()}): {score_bot1} Trận thắng")
    print(f"🏆 [{name_2}] ({args.mode2.upper()}): {score_bot2} Trận thắng")
    print(f"🤝 Số trận HÒA: {draws}")
    print("=====================================================")

if __name__ == "__main__":
    start_arena()