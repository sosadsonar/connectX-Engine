# play.py
import json
import os
import time
from board import ConnectXBoard
from ai import AdvancedNegamaxAI

def render_board(board: ConnectXBoard):
    """Giải mã Bitboard phẳng để vẽ giao diện trực quan ra màn hình"""
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

def main():
    w, h, x = 7, 6, 4      
    max_depth = 20          
    time_limit = 2.2        

    print("\n" + "*"*50)
    print("🏆 KHAI MẠC PHÒNG ĐẤU: NGƯỜI VS SIÊU TRÍ TUỆ NHÂN TẠO HEURISTIC 🏆")
    print(f"   Cấu hình luật đấu: Connect {x} trên bàn cờ {w}x{h}")
    print("*"*50)

    ai_mode = "heuristic"

    # Nạp trọng số Heuristic
    weights = {}
    weight_path = f"../models/checkpoint_7x6_x4_gen_15.json"
    if os.path.exists(weight_path):
        with open(weight_path, "r") as f:
            weights = json.load(f)
        print(f"\n[OK] Đã nạp thành công bộ não HCE tinh anh từ: '{weight_path}'")
    else:
        try:
            from train import BASE_CHAMPION_WEIGHTS
            weights = BASE_CHAMPION_WEIGHTS
            print("\n[⚠️ WARNING] Không tìm thấy file JSON, dùng não sơ sinh HCE mặc định!")
        except ImportError:
            print("\n[❌ LỖI] Không tìm thấy cấu hình HCE nào!")
            return

    # Khởi tạo bàn cờ bitboard
    board = ConnectXBoard(w, h, x)
    
    # Lựa chọn lượt đi
    choice = input("\nBạn muốn cầm quân nào? (0: Đi trước [X],  1: Đi sau [O]): ").strip()
    human_id = int(choice) if choice in ['0', '1'] else 0
    bot_id = 1 - human_id
    
    # Khởi tạo Bot chạy Heuristic thuần
    bot = AdvancedNegamaxAI(weights, player_id=bot_id, mode=ai_mode, tt_exponent=23)
    
    current_player = 0
    render_board(board)
    
    while board.get_valid_cols():
        if current_player == human_id:
            valid_cols = board.get_valid_cols()
            try:
                move = int(input(f"\n>> Lượt của bạn ({'X' if human_id == 0 else 'O'}). Nhập số cột {valid_cols}: "))
                if move not in valid_cols:
                    print("❌ Cột đã đầy hoặc nằm ngoài bàn cờ! Hãy chọn lại.")
                    continue
            except ValueError:
                print("❌ Vui lòng chỉ nhập số nguyên hợp lệ!")
                continue
        else:
            print(f"\n💻 Bot [HEURISTIC] đang dệt lưới tính toán... 🤔")
            start_think = time.perf_counter()
            
            move = bot.select_move(board, max_depth=max_depth, time_limit=time_limit, verbose=True)
            
            end_think = time.perf_counter()
            print(f"⚡ Bot quyết định hạ quân vào cột [{move}] (Thời gian nghĩ: {end_think - start_think:.3f} giây)")
            
        board.make_move(move, current_player)
        render_board(board)
        
        if board.check_win(current_player):
            if current_player == human_id:
                print(f"\n🎉 KINH NGẠC! Bạn đã đánh bại Siêu Bot!")
            else:
                print(f"\n💀 GAME OVER! Bot đã gài bẫy và hủy diệt bạn hoàn toàn. Hãy phục thù ở ván sau!")
            return
            
        current_player = 1 - current_player
        
    print("\n🤝 Bàn cờ đã kín! Trận đấu kết thúc với kết quả HÒA CĂNG THẲNG.")

if __name__ == "__main__":
    main()