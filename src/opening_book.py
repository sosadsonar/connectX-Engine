# opening_book.py
import struct
import math
from board import ConnectXBoard

def is_prime(n: int) -> bool:
    if n < 2: return False
    for i in range(2, int(math.isqrt(n)) + 1):
        if n % i == 0: return False
    return True

def next_prime(n: int) -> int:
    while not is_prime(n):
        n += 1
    return n

class PythonOpeningBookReader:
    def __init__(self, book_path="./7x6.book"):
        self.book_path = book_path
        self.depth = -1
        self.width = 7
        self.height = 6
        self.size = 0
        self.partial_key_bytes = 0
        
        self.keys = []
        self.values = []
        self.is_loaded = False
        
        if book_path:
            self.load_book(book_path)

    def load_book(self, filename: str):
        try:
            with open(filename, "rb") as f:
                header = f.read(6)
                if len(header) < 6: return
                
                _width, _height, _depth, _partial_key_bytes, _value_bytes, _log_size = struct.unpack("6b", header)
                if _width != self.width or _height != self.height: return

                self.depth = _depth
                self.partial_key_bytes = _partial_key_bytes
                
                base_size = 1 << _log_size
                self.size = next_prime(base_size)
                
                key_bytes = f.read(self.size * self.partial_key_bytes)
                fmt_map = {1: "B", 2: "H", 4: "I", 8: "Q"}
                self.keys = list(struct.unpack(f"{self.size}{fmt_map[self.partial_key_bytes]}", key_bytes))
                
                value_bytes = f.read(self.size * _value_bytes)
                self.values = list(struct.unpack(f"{self.size}B", value_bytes))
                self.is_loaded = True
                print(f"📖 [NATIVE BOOK] Đã nạp thành công '{filename}' (Depth: {self.depth}, Size: {self.size} entries)")
        except Exception as e:
            print(f"❌ Không thể đọc file opening book trực tiếp: {e}")

    def _partialKey3(self, key: int, current_position: int, mask: int, col: int) -> int:
        """
        MÔ PHỎNG CHUẨN 100% C++ partialKey3:
        Tích lũy giá trị cơ số 3 cho một cột cụ thể.
        """
        idx = col * 7  # col_height luôn cố định là 7 với bàn cờ 7x6
        while (mask & (1 << idx)) != 0:
            key *= 3
            if (current_position & (1 << idx)) != 0:
                key += 1
            else:
                key += 2
            idx += 1
        key *= 3
        return key

    def _compute_key3(self, board: ConnectXBoard, current_player_id: int) -> int:
        """
        Tính toán mã băm key3 đối xứng gương phẳng tự nhiên.
        """
        current_position = board.boards[current_player_id]
        mask = board.boards[0] | board.boards[1]

        # 1. CHIỀU XUÔI: Duyệt col tăng dần từ 0 đến 6
        key_forward = 0
        for c in range(self.width):
            key_forward = self._partialKey3(key_forward, current_position, mask, c)

        # 2. CHIỀU NGƯỢC: Duyệt col giảm dần từ 6 về 0
        key_reverse = 0
        for c in range(self.width - 1, -1, -1):
            key_reverse = self._partialKey3(key_reverse, current_position, mask, c)

        # Logic so sánh nhỏ hơn (<) gốc của Pascal Pons
        if key_forward < key_reverse:
            return key_forward // 3
        else:
            return key_reverse // 3

    def get_score_from_book(self, board: ConnectXBoard, current_player_id: int) -> int:
        if not self.is_loaded: return None
        
        move_count = board.boards[0].bit_count() + board.boards[1].bit_count()
        if move_count > self.depth: return None 

        k3 = self._compute_key3(board, current_player_id)
        idx = k3 % self.size
        
        mask_bits = (1 << (self.partial_key_bytes * 8)) - 1
        truncated_k3 = k3 & mask_bits
        
        if self.keys[idx] == truncated_k3:
            raw_val = self.values[idx]
            if raw_val == 0: return None
            return raw_val + (-(self.width * self.height) // 2 + 3) - 1
        return None