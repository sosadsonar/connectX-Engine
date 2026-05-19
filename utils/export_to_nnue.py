# export_to_numpy.py
import os
import torch
import numpy as np

def export_pytorch_to_numpy():
    # Cấu hình đường dẫn file đầu vào và đầu ra
    model_path = "../nnue/best_nnue_model (1).pt"
    output_path = "../nnue/nnue_weights (1).npz"

    if not os.path.exists(model_path):
        print(f"❌ [LỖI] Không tìm thấy file '{model_path}' trong thư mục hiện tại!")
        print("👉 Ông hãy tải file này từ Google Drive (thư mục nnue) về đặt cạnh file script này nhé.")
        return

    print(f"📦 Đang nạp bộ não PyTorch từ tệp: '{model_path}'...")
    
    # Ép nạp lên CPU để chạy mượt mà trên mọi máy tính (kể cả máy nhà ông không có card rời)
    state_dict = torch.load(model_path, map_location="cpu")

    # Trích xuất và chuyển vế (Transpose) để khớp hoàn hảo với công thức: inputs @ W + b
    numpy_weights = {
        "W1": state_dict["fc1.weight"].numpy().T,
        "b1": state_dict["fc1.bias"].numpy(),
        "W2": state_dict["fc2.weight"].numpy().T,
        "b2": state_dict["fc2.bias"].numpy(),
        "W3": state_dict["fc3.weight"].numpy().T,
        "b3": state_dict["fc3.bias"].numpy()
    }

    print("💾 Đang nén dải ma trận phẳng và đóng gói xuống ổ cứng...")
    # Lưu dưới dạng tệp nén nhị phân siêu nhẹ của NumPy (~76KB)
    np.savez(output_path, **numpy_weights)

    print("\n=====================================================")
    print(" 🏆 CHIẾN DỊCH CHUYỂN HỆ ĐẦU NÃO THÀNH CÔNG RỰC RỠ!")
    print("=====================================================")
    print(f" Tệp nhị phân xuất xưởng: '{output_path}'")
    print(f" 📊 Chi tiết thông số kỹ thuật bộ não V1.0:")
    print(f"   - Tầng ẩn 1 (W1): {numpy_weights['W1'].shape} | Bias 1 (b1): {numpy_weights['b1'].shape}")
    print(f"   - Tầng ẩn 2 (W2): {numpy_weights['W2'].shape} | Bias 2 (b2): {numpy_weights['b2'].shape}")
    print(f"   - Tầng Ra   (W3): {numpy_weights['W3'].shape} | Bias 3 (b3): {numpy_weights['b3'].shape}")
    print("=====================================================")
    print("🔥 Giờ ông chỉ việc nộp file 'nnue_weights.npz' này kèm file 'ai.py' lên Kaggle là Bot chạy xé gió!")

if __name__ == "__main__":
    export_pytorch_to_numpy()