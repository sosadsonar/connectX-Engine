class TranspositionTable:
    def __init__(self, exponent=23):
        """
        Khởi tạo Bảng chuyển vị (Transposition Table) tĩnh trong bộ nhớ.
        - exponent: Số mũ cơ số 2 để quyết định kích thước bảng (mặc định 21 -> 2^21 entries).
        """
        self.exponent = exponent
        self.entries = 1 << exponent  # Phép dịch bit tương đương 2^exponent
        self.mask = self.entries - 1  # Mặt nạ bit để lấy phần dư (index) siêu tốc thay cho phép chia %
        self.table = [None] * self.entries

    def lookup(self, zobrist_key):
        """
        Tra cứu trạng thái bàn cờ hiện tại trong bảng băm với độ phức tạp O(1).
        Trả về mảng 5 trường dữ liệu nếu khớp key, ngược lại trả về None.
        """
        idx = zobrist_key & self.mask
        entry = self.table[idx]
        
        if entry and entry[0] == zobrist_key:
            return entry
        return None

    def store(self, zobrist_key, depth, flag, score, best_move):
        """
        Lưu trạng thái thế cờ vừa tính toán xong vào bảng băm kèm theo nước đi gợi ý.
        Áp dụng chiến lược thay thế thông minh (Depth-Preferred Replacement).
        """
        idx = zobrist_key & self.mask
        existing = self.table[idx]
        if existing is None or depth >= existing[1]:
            self.table[idx] = (zobrist_key, depth, flag, score, best_move)
