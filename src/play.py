# play.py
import json
import os
import time
from board import ConnectXBoard
from ai import AdvancedNegamaxAI

def render_board(board: ConnectXBoard):
    """Giải mã Bitboard phẳng của bạn để vẽ giao diện trực quan ra màn hình"""
    print("\n" + "=" * 25)
    # Quét từ hàng cao nhất xuống hàng 0
    for row in range(board.h - 1, -1, -1):
        row_str = "| "
        for col in range(board.w):
            # Công thức dịch bit dựa trên cấu hình flat bitboard của bạn
            idx = col * board.col_height + row
            mask = 1 << idx
            
            if board.boards[0] & mask:
                row_str += "X  "  # Người chơi 0 (Quân X)
            elif board.boards[1] & mask:
                row_str += "O  "  # Người chơi 1 (Quân O)
            else:
                row_str += ".  "  # Ô trống
        row_str += "|"
        print(row_str)
        
    print("-" * (board.w * 3 + 4))
    # In số thứ tự cột ở dưới đáy để bạn dễ nhìn và chọn
    print("  " + "  ".join(str(i) for i in range(board.w)))
    print("=" * 25)

def main():
    # --- CẤU HÌNH PHÒNG ĐẤU ---
    w, h, x = 7, 6, 4      
    max_depth = 20          # Cho phép lặn sâu hơn chút vì code mới đã cực nhanh
    time_limit = 1.8        

    print("\n" + "*"*50)
    print("🏆 KHAI MẠC PHÒNG ĐẤU: NGƯỜI VS SIÊU TRÍ TUỆ NHÂN TẠO 🏆")
    print(f"   Cấu hình luật đấu: Connect {x} trên bàn cờ {w}x{h}")
    print("*"*50)

    # 1. MENU CHỌN CHẾ ĐỘ AI
    print("\nBạn muốn khiêu chiến với phiên bản AI nào?")
    print("  [1] HEURISTIC (Công thức toán học - Đánh bài bản, bảo thủ)")
    print("  [2] NNUE      (Mạng Nơ-ron thuần - Đánh linh hoạt, quái dị)")
    print("  [3] HYBRID    (Quái vật lai ghép - Trực giác NNUE + Sát thủ Heuristic) [KHUYÊN DÙNG]")
    
    ai_choice = input("Nhập lựa chọn của bạn (1/2/3): ").strip()
    if ai_choice == '1':
        ai_mode = "heuristic"
    elif ai_choice == '2':
        ai_mode = "nnue"
    else:
        ai_mode = "hybrid" # Mặc định là Hybrid nếu nhập sai

    # 2. Nạp trọng số Heuristic (Cần thiết cho Heuristic và Hybrid)
    weights = {}
    if ai_mode in ["heuristic", "hybrid"]:
        weight_path = f"../models/best_weights_7x6_x4.json"
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
    
    if ai_mode in ["nnue", "hybrid"]:
        print(f"[OK] Mạng Nơ-ron (NNUE) đã được kích hoạt sẵn sàng chiến đấu!")

    # 3. Khởi tạo bàn cờ bitboard
    board = ConnectXBoard(w, h, x)
    
    # 4. Lựa chọn lượt đi
    choice = input("\nBạn muốn cầm quân nào? (0: Đi trước [X],  1: Đi sau [O]): ").strip()
    human_id = int(choice) if choice in ['0', '1'] else 0
    bot_id = 1 - human_id
    
    # Khởi tạo Bot với chế độ đã chọn
    bot = AdvancedNegamaxAI(weights, player_id=bot_id, mode=ai_mode, tt_exponent=23)
    
    current_player = 0
    render_board(board)
    
    # 5. Vòng lặp trận đấu
    while board.get_valid_cols():
        if current_player == human_id:
            # --- LƯỢT CỦA BẠN ---
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
            # --- LƯỢT CỦA BOT ---
            print(f"\n💻 Bot [{ai_mode.upper()}] đang dệt lưới tính toán... 🤔")
            start_think = time.perf_counter()
            
            # Bot gọi AI
            move = bot.select_move(board, max_depth=max_depth, time_limit=time_limit, verbose=True)
            
            end_think = time.perf_counter()
            print(f"⚡ Bot quyết định hạ quân vào cột [{move}] (Thời gian nghĩ: {end_think - start_think:.3f} giây)")
            
        # Thực hiện nước đi trên Bitboard
        board.make_move(move, current_player)
        render_board(board)
        
        # Kiểm tra thắng cuộc
        if board.check_win(current_player):
            if current_player == human_id:
                print(f"\n🎉 KINH NGẠC! Bạn đã đánh bại Siêu Bot [{ai_mode.upper()}] do chính tay mình luyện ra!")
            else:
                print(f"\n💀 GAME OVER! Bot [{ai_mode.upper()}] đã gài bẫy và hủy diệt bạn hoàn toàn. Hãy phục thù ở ván sau!")
            return
            
        current_player = 1 - current_player
        
    print("\n🤝 Bàn cờ đã kín! Trận đấu kết thúc với kết quả HÒA CĂNG THẲNG.")

if __name__ == "__main__":
    main()