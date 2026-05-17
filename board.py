import random

class ConnectXBoard:
    def __init__(self, w=7, h=6, x=4):
        self.w = w          # Chiều rộng
        self.h = h          # Chiều cao thực tế
        self.x = x          # Số quân liên tiếp để thắng (Connect X)
        self.col_height = h + 1  # Chiều cao tính cả hàng đệm
        
        # 2 bảng bitboard riêng biệt: 0 cho Người chơi A, 1 cho Người chơi B
        self.boards = [0, 0] 
        self.heights = [c * self.col_height for c in range(self.w)]
        
        # Mặt nạ để loại bỏ hàng đệm khi cần tính toán ô trống
        self.valid_mask = 0
        for c in range(self.w):
            for r in range(self.h):
                self.valid_mask |= (1 << (c * self.col_height + r))
                
        # 4 hướng dịch bit cơ bản
        self.shifts = [1, self.col_height, self.col_height + 1, self.col_height - 1]
        
        self.center_order = sorted(range(self.w), key=lambda c: abs(c - self.w / 2))
        max_cells = self.w * self.col_height
        rng = random.Random(42)  # Seed cố định để bảo toàn tính nhất quán khi tra bảng băm
        
        # Sinh mảng phẳng 2 chiều chứa các số ngẫu nhiên 64-bit cho từng vị trí ô cờ
        self.zobrist_table = [
            [rng.getrandbits(64) for _ in range(max_cells)],
            [rng.getrandbits(64) for _ in range(max_cells)]
        ]
        self.zobrist_turn = rng.getrandbits(64)
        self.zobrist_key = 0  # Bàn cờ rỗng khởi tạo bằng mã 0

    def make_move(self, col, player_id):
        """Đặt quân cờ vào cột col và cập nhật gối đầu mã băm Zobrist Key"""
        idx = self.heights[col]
        self.boards[player_id] |= (1 << idx)
        self.zobrist_key ^= self.zobrist_table[player_id][idx]
        self.zobrist_key ^= self.zobrist_turn
        
        self.heights[col] += 1

    def undo_move(self, col, player_id):
        """Rút quân cờ và hoàn tác Zobrist Key về trạng thái trước đó trong O(1)"""
        self.heights[col] -= 1
        idx = self.heights[col]
        self.boards[player_id] &= ~(1 << idx)
        self.zobrist_key ^= self.zobrist_table[player_id][idx]
        self.zobrist_key ^= self.zobrist_turn

    def get_valid_cols(self):
        """Trả về danh sách các cột hợp lệ theo thứ tự tối ưu từ giữa ra rìa"""
        return [c for c in self.center_order if (self.heights[c] % self.col_height) < self.h]

    def check_win(self, player_id):
        """Kiểm tra chiến thắng bằng Gập nhị phân - Tự động thích ứng với mọi X"""
        board = self.boards[player_id]
        for shift in self.shifts:
            filled = board
            len_filled = 1
            while len_filled < self.x:
                delta = min(len_filled, self.x - len_filled)
                filled &= (filled >> (delta * shift))
                len_filled += delta
            if filled != 0:
                return True
        return False