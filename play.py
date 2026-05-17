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
    # --- CẤU HÌNH PHÒNG ĐẤU (Hãy sửa đúng theo file bạn vừa train) ---
    w, h, x = 10, 10, 5      # Kích thước bàn cờ muốn test (Ví dụ: 7x6, X=4 hoặc 10x12, X=5)
    max_depth = 12          # Độ sâu giới hạn cho Bot (IDS sẽ tự động điều tốc)
    time_limit = 1.8        # Giới hạn thời gian Bot nghĩ (1.5 giây để bạn đỡ phải đợi lâu)

    # 1. Tự động tìm kiếm bộ não đã tối ưu vĩnh viễn
    weight_path = f"models/checkpoint_7x6_x4_gen_10.json"
    if os.path.exists(weight_path):
        with open(weight_path, "r") as f:
            weights = json.load(f)
        print(f"[OK] Đã nạp thành công bộ não tinh anh từ: '{weight_path}'")
    else:
        # Nếu chưa có file train, lấy tạm cấu hình mặc định trong train.py
        from train import BASE_CHAMPION_WEIGHTS
        weights = BASE_CHAMPION_WEIGHTS
        print("[⚠️ WARNING] Không tìm thấy file trọng số đã train, Bot tạm thời dùng não sơ sinh!")

    # 2. Khởi tạo bàn cờ bitboard
    board = ConnectXBoard(w, h, x)
    
    print("\n" + "*"*45)
    print(" KHAI MẠC PHÒNG ĐẤU NGƯỜI VS BOT BISEWISE CNN")
    print(f" Cấu hình luật đấu: Connect {x} trên bàn cờ {w}x{h}")
    print("*"*45)
    
    # 3. Lựa chọn lượt đi
    choice = input("Bạn muốn đi trước hay đi sau? (Gõ 0 để đi trước, gõ 1 để đi sau): ")
    human_id = int(choice) if choice in ['0', '1'] else 0
    bot_id = 1 - human_id
    
    # Khởi tạo Bot với ID phe đối lập
    bot = AdvancedNegamaxAI(weights, player_id=bot_id, tt_exponent=21)
    
    current_player = 0
    render_board(board)
    
    # 4. Vòng lặp trận đấu
    while board.get_valid_cols():
        if current_player == human_id:
            # --- LƯỢT CỦA BẠN ---
            valid_cols = board.get_valid_cols()
            print(f"Các cột bạn có thể đi: {valid_cols}")
            try:
                move = int(input(f"Lượt của bạn ({'X' if human_id == 0 else 'O'}). Nhập số cột: "))
                if move not in valid_cols:
                    print("Cột đã đầy hoặc nằm ngoài bàn cờ! Hãy chọn lại.")
                    continue
            except ValueError:
                print("Vui lòng chỉ nhập số nguyên hợp lệ!")
                continue
        else:
            # --- LƯỢT CỦA BOT ---
            print("Bot đang dệt lưới tính toán... 🤔")
            start_think = time.time()
            
            # Bot gọi IDS kết hợp Timeout dưới 2s
            move = bot.select_move(board, max_depth=max_depth, time_limit=time_limit)
            
            end_think = time.time()
            print(f"Bot hạ quân vào cột [{move}] (Thời gian nghĩ: {end_think - start_think:.2f} giây)")
            
        # Thực hiện nước đi trên Bitboard
        board.make_move(move, current_player)
        render_board(board)
        
        # Kiểm tra thắng cuộc bằng hàm nhị phân siêu tốc
        if board.check_win(current_player):
            if current_player == human_id:
                print("\n🎉 Kinh ngạc chưa! Bạn đã đánh bại con Bot siêu cấp do chính mình luyện ra!")
            else:
                print("\n💀 Bot đã gài bẫy thành công và hủy diệt bạn! Hãy cắm máy train sâu hơn đi.")
            return
            
        current_player = 1 - current_player
        
    print("\n🤝 Bàn cờ đã đầy! Trận đấu kết thúc với kết quả Hòa.")

if __name__ == "__main__":
    main()
