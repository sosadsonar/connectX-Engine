# nnue_eval.py
import os
import numpy as np
from board import ConnectXBoard

class NNUEEvaluator:
    def __init__(self, model_path: str, board_w: int, board_h: int):
        """Bộ lượng giá Mạng Nơ-ron Độc lập - Tự động thích ứng không gian Nhị phân"""
        self.w = board_w
        self.h = board_h
        self.board_cells = self.w * self.h
        self.input_size = self.board_cells * 2  # Gấp đôi vì có 2 mảng bit của Ta và Địch

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ [NNUE LỖI CHÍ MẠNG] Không tìm thấy tệp trọng số mạng Nơ-ron tại: '{model_path}'")

        # Nạp ma trận vào RAM
        with np.load(model_path) as data:
            self.W1, self.b1 = data["W1"], data["b1"]
            self.W2, self.b2 = data["W2"], data["b2"]
            self.W3, self.b3 = data["W3"], data["b3"]
            # 🎯 Đọc Scale tự động (Nếu không có thì dùng mặc định 13194)
            self.scale_k = float(data["scale_k"][0])
        
        # 🛡️ SAFETY CHECK: Rào chắn bảo vệ không gian ma trận
        # Đảm bảo mạng được train cho đúng kích thước bàn cờ hiện tại
        if self.W1.shape[0] != self.input_size:
            raise ValueError(
                f"❌ [NNUE XUNG ĐỘT KÍCH THƯỚC] Bàn cờ hiện tại là {self.w}x{self.h} (Input {self.input_size} chiều), "
                f"nhưng mạng Nơ-ron được nạp lại yêu cầu Input {self.W1.shape[0]} chiều. "
                f"Vui lòng train một bộ não NNUE riêng cho bàn cờ {self.w}x{self.h}!"
            )

        # 🚀 Tự động tính toán mảng dịch bit (Shifts) tương thích với cấu trúc Bitboard
        col_height = self.h + 1
        self.shifts = np.array([c * col_height + r for c in range(self.w) for r in range(self.h)], dtype=np.uint64)

        # Cấp phát sẵn bộ nhớ tĩnh (Zero-Allocation Buffer) để tăng tốc độ Search
        self.input_buffer = np.zeros(self.input_size, dtype=np.float32)
        self.h1_buffer = np.zeros(self.W1.shape[1], dtype=np.float32)
        self.h2_buffer = np.zeros(self.W2.shape[1], dtype=np.float32)

    def evaluate(self, board: ConnectXBoard, current_player_id: int) -> int:
        """Thực thi suy luận Neural Network cực tốc bằng Numpy C-Backend"""
        us_mask = board.boards[current_player_id]
        them_mask = board.boards[1 - current_player_id]
        
        # 1. Trích xuất Vector đặc trưng
        self.input_buffer[:self.board_cells] = (us_mask >> self.shifts) & 1
        self.input_buffer[self.board_cells:] = (them_mask >> self.shifts) & 1
        
        # 2. Phóng qua Tầng Ẩn 1 (Hidden Layer 1)
        np.dot(self.input_buffer, self.W1, out=self.h1_buffer)
        self.h1_buffer += self.b1
        np.clip(self.h1_buffer, 0.0, 1.0, out=self.h1_buffer) 
        
        # 3. Phóng qua Tầng Ẩn 2 (Hidden Layer 2)
        np.dot(self.h1_buffer, self.W2, out=self.h2_buffer)
        self.h2_buffer += self.b2
        np.clip(self.h2_buffer, 0.0, 1.0, out=self.h2_buffer) 
        
        # 4. Xuất kết quả (Output Layer)
        output = np.dot(self.h2_buffer, self.W3) + self.b3
        
        # 🎯 Khôi phục điểm số tự động và siêu mượt
        return int(output[0] * self.scale_k)