# inspect_data.py
import numpy as np

def inspect_npy_file(file_path, num_samples_to_show=3):
    print(f"=====================================================")
    print(f"🔍 ĐANG KIỂM TRA FILE: {file_path}")
    print(f"=====================================================\n")

    # 1. Nạp file (Bắt buộc phải có allow_pickle=True và .item())
    try:
        data = np.load(file_path, allow_pickle=True).item()
    except Exception as e:
        print(f"[LỖI] Không thể đọc file: {e}")
        return

    # 2. Kiểm tra các trường dữ liệu (Keys)
    print(f"🔹 Các trường dữ liệu có trong file: {list(data.keys())}")
    
    # 3. Xem cấu hình sa bàn (Meta Board)
    meta = data.get("meta_board", [7, 6, 4])
    w, h, x = int(meta[0]), int(meta[1]), int(meta[2])
    print(f"🔹 Cấu hình bàn cờ: {w}x{h} (Luật Connect {x})\n")

    # 4. Kiểm tra kích thước (Shape) và Kiểu dữ liệu (Dtype)
    print("🔹 Thông tin các mảng NumPy:")
    for key in ["us_mask", "them_mask", "search_score", "game_result"]:
        if key in data:
            # Ép kiểu str() cho shape để f-string căn lề chuẩn xác
            shape_str = str(data[key].shape)
            print(f"   - {key:13}: Shape = {shape_str:<10} | Dtype = {data[key].dtype}")
    print("-" * 53)

    # Helper function để vẽ bàn cờ trực quan từ bitmask (Chuẩn 7x6 John Tromp)
    def draw_board_from_masks(us_mask, them_mask):
        col_height = h + 1  # Cộng 1 dummy row theo cấu trúc Bitboard của bạn
        # Tạo lưới trống h hàng x w cột
        grid = [["." for _ in range(w)] for _ in range(h)]
        for c in range(w):
            for r in range(h):
                bit_idx = c * col_height + r
                if (us_mask >> bit_idx) & 1:
                    grid[h - 1 - r][c] = "X"   # X là quân ta
                elif (them_mask >> bit_idx) & 1:
                    grid[h - 1 - r][c] = "O"   # O là quân địch
        # In bàn cờ ra màn hình
        for row in grid:
            print("  " + " ".join(row))

    # 5. In chi tiết một vài mẫu đầu tiên
    total_samples = len(data["search_score"])
    num_samples_to_show = min(num_samples_to_show, total_samples)
    
    print(f"\n🔹 Hiển thị ngẫu nhiên {num_samples_to_show} mẫu dữ liệu đầu tiên:")
    for i in range(num_samples_to_show):
        print(f"\n👉 [MẪU SỐ {i+1}] (Index {i})")
        print(f"   • Us Mask (Decimal)  : {data['us_mask'][i]}")
        print(f"   • Them Mask (Decimal): {data['them_mask'][i]}")
        print(f"   • Search Score       : {data['search_score'][i]:+d}")
        print(f"   • Game Result        : {data['game_result'][i]} (1.0=Thắng, 0.5=Hòa, 0.0=Thua)")
        print("   • Hình ảnh sa bàn thực tế:")
        draw_board_from_masks(data['us_mask'][i], data['them_mask'][i])
        print("   " + "~" * 30)

if __name__ == "__main__":
    # Điền đường dẫn tới file bạn muốn kiểm tra vào đây
    inspect_npy_file("../data/final_nnue_train.npy", num_samples_to_show=20)
