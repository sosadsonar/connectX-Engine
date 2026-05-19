# nnue_train.py
import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class ConnectXNNUE(nn.Module):
    """Kiến trúc mạng NNUE ConnectX nâng cấp: 84 -> 128 -> 64 -> 1 (Dùng Clipped ReLU)"""
    def __init__(self, input_size=84, hidden1=128, hidden2=64):
        super(ConnectXNNUE, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)

    def forward(self, x):
        # Sử dụng Clipped ReLU (giới hạn từ 0.0 đến 1.0) giúp NumPy tăng tốc lúc Search
        x = torch.clamp(self.fc1(x), min=0.0, max=1.0)
        x = torch.clamp(self.fc2(x), min=0.0, max=1.0)
        x = self.fc3(x) 
        return x

class ConnectXDataset(Dataset):
    """Bộ nạp và giải mã Bitmask nhị phân của tệp .npy"""
    def __init__(self, file_path):
        print(f"[HỆ THỐNG DATA] Đang đọc tệp nhị phân dữ liệu: '{file_path}'...")
        data = np.load(file_path, allow_pickle=True).item()
        
        meta = data.get("meta_board", np.array([7, 6, 4], dtype=np.int32))
        self.w, self.h, self.x = int(meta[0]), int(meta[1]), int(meta[2])
        
        us_masks = data["us_mask"]
        them_masks = data["them_mask"]
        self.scores = torch.tensor(data["search_score"], dtype=torch.float32)
        self.results = torch.tensor(data["game_result"], dtype=torch.float32)
        
        num_samples = len(us_masks)
        print(f"[HỆ THỐNG DATA] Phát hiện cấu hình bàn cờ: {self.w}x{self.h} (Luật Connect {self.x})")
        print(f"[HỆ THỐNG DATA] Tổng số mẫu tích lũy: {num_samples:,} thế cờ.")
        print(f"[HỆ THỐNG DATA] Đang giải nén Bitmask sang Vector phẳng {self.w * self.h * 2} chiều...")
        
        self.inputs = np.zeros((num_samples, self.w * self.h * 2), dtype=np.float32)
        idx = 0
        col_height = self.h + 1
        for c in range(self.w):
            for r in range(self.h):
                bit_idx = c * col_height + r
                self.inputs[:, idx] = (us_masks >> bit_idx) & 1
                self.inputs[:, idx + (self.w * self.h)] = (them_masks >> bit_idx) & 1
                idx += 1
                
        self.inputs = torch.tensor(self.inputs, dtype=torch.float32)
        print("[HỆ THỐNG DATA] Đường ống giải nén hoàn tất! Dữ liệu đã sẵn sàng nạp vào RAM.\n")

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        return self.inputs[idx], self.scores[idx], self.results[idx]

class StockfishLoss(nn.Module):
    """Lõi tính toán sai số trộn nhãn chiến thuật kiểu Stockfish"""
    def __init__(self, lambda_blend=0.5, scale_k=13194.0, power=2.6):
        super(StockfishLoss, self).__init__()
        self.lambda_blend = lambda_blend
        self.scale_k = scale_k
        self.power = power

    def forward(self, pred_score, target_score, target_result):
        wdl_target = torch.sigmoid(target_score / self.scale_k)
        blended_target = self.lambda_blend * wdl_target + (1.0 - self.lambda_blend) * target_result
        pred_wdl = torch.sigmoid(pred_score.squeeze(-1))
        loss = torch.mean(torch.abs(pred_wdl - blended_target) ** self.power)
        return loss

def export_to_numpy(model, filepath, scale_k):
    """Rút ruột PyTorch Model và xuất ra ma trận NumPy tĩnh cho engine thực chiến"""
    weights = model.state_dict()
    
    # PyTorch lưu Linear Weight dưới dạng [out_features, in_features]
    # np.dot() trong Engine yêu cầu [in_features, out_features] nên PHẢI có .T (Transpose)
    np.savez(
        filepath,
        W1=weights['fc1.weight'].cpu().numpy().T,
        b1=weights['fc1.bias'].cpu().numpy(),
        W2=weights['fc2.weight'].cpu().numpy().T,
        b2=weights['fc2.bias'].cpu().numpy(),
        W3=weights['fc3.weight'].cpu().numpy().T,
        b3=weights['fc3.bias'].cpu().numpy(),
        # Nhúng Scale_K vào đây để NNUE Eval dùng
        scale_k=np.array([scale_k], dtype=np.float32) 
    )
    print(f"    📦 [EXPORT] Đã đóng gói ma trận C-NumPy (kèm Scale={scale_k}) vào: '{filepath}'")

def train_nnue():
    parser = argparse.ArgumentParser(description="ConnectX NNUE Custom Stockfish Trainer with Resume Capability")
    parser.add_argument("--train_data", type=str, default="../data/final_nnue_train.npy", help="Đường dẫn file dữ liệu Train sạch")
    parser.add_argument("--val_data", type=str, default="../data/final_nnue_val.npy", help="Đường dẫn file dữ liệu Validation bảo hiểm")
    parser.add_argument("--epochs", type=int, default=150, help="Số lượng kỷ nguyên huấn luyện")
    parser.add_argument("--batch_size", type=int, default=1024, help="Kích thước lô dữ liệu (Batch Size)")
    parser.add_argument("--lr", type=float, default=0.0001, help="Tốc độ học khởi điểm (Learning Rate)")
    parser.add_argument("--lambda_blend", type=float, default=0.5, help="Tỷ lệ trộn Lambda")
    parser.add_argument("--power", type=float, default=2.6, help="Số mũ của hàm Loss nâng cao Stockfish")
    parser.add_argument("--out", type=str, default="../nnue/best_nnue_model.pt", help="Tên file lưu mô hình tốt nhất")
    parser.add_argument("--ckpt_interval", type=int, default=10, help="Chu kỳ lưu checkpoint bảo hiểm (số epoch)")
    parser.add_argument("--resume", type=str, default="", help="Đường dẫn tới file .ckpt để train tiếp tục nếu máy bị sập")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Hệ số phạt L2 Regularization chống overfit")
    
    # Tham số cấu hình SCALE
    parser.add_argument("--scale_k", type=float, default=None, help="Hệ số nén Sigmoid (Scale K). Nếu không nhập, đọc từ JSON.")
    parser.add_argument("--hce_config", type=str, default="../models/best_weights_7x6_x4.json", help="File JSON của não HCE để đồng bộ Scale")
    args = parser.parse_args()

    # ---- 🚀 CƠ CHẾ XÁC ĐỊNH SCALE_K SIÊU CHẶT CHẼ ----
    dynamic_scale_k = None

    if args.scale_k is not None:
        dynamic_scale_k = args.scale_k
        print(f"🎯 [MANUAL SCALE] Sử dụng Scale_K = {dynamic_scale_k} từ thiết lập dòng lệnh!")
    else:
        try:
            with open(args.hce_config, "r") as f:
                hce_weights = json.load(f)
            if "MAX_STRATEGIC" in hce_weights:
                dynamic_scale_k = float(hce_weights["MAX_STRATEGIC"])
                print(f"🎯 [AUTO SCALE] Đã đồng bộ Scale_K = {dynamic_scale_k} từ '{args.hce_config}'!")
            else:
                raise KeyError("File JSON không có trường 'MAX_STRATEGIC'.")
        except Exception as e:
            print(f"\n❌ [LỖI SCALE_K CHÍ MẠNG] Không thể xác định Scale_K tự động. Lý do: {e}")
            print(f"👉 Hãy truyền trực tiếp qua dòng lệnh bằng cờ: --scale_k <con_số>")
            print(f"Ví dụ: python nnue_train.py --scale_k 13194.0\n")
            return  # Dừng chương trình

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n=====================================================")
    print(" 🔥 KÍCH HOẠT LÒ LUYỆN MA TRẬN MẠNG NNUE CHỐNG RÒ RỈ V2.0")
    print(f" Thiết bị phần cứng đang gánh tải: {device} 🚀")
    print(f" Tự động lưu file phục hồi tiến trình sau mỗi: {args.ckpt_interval} Epoch 🛡️")
    print(f" Cấu hình tối ưu: Weight Decay = {args.weight_decay} | Tích hợp bộ điều tốc LR Scheduler 📈")
    print("=====================================================\n")

    train_dataset = ConnectXDataset(args.train_data)
    val_dataset = ConnectXDataset(args.val_data)
    
    # Bật Đa luồng (num_workers) để GPU không bị đói data
    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = ConnectXNNUE(input_size=train_dataset.w * train_dataset.h * 2).to(device)
    criterion = StockfishLoss(lambda_blend=args.lambda_blend, scale_k=dynamic_scale_k, power=args.power)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5)

    start_epoch = 1
    best_val_loss = float('inf')
    checkpoint_path = f"{args.out}.ckpt"

    resume_path = args.resume if args.resume else checkpoint_path
    if resume_path and os.path.exists(resume_path):
        print(f"[HỆ THỐNG] Phát hiện file checkpoint '{resume_path}'. Đang khôi phục trạng thái cũ...")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"▶️ Khôi phục thành công! Sẽ tiếp tục cày từ Epoch {start_epoch} (Best Val Loss cũ: {best_val_loss:.6f})")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss = 0.0
        
        for inputs, scores, results in train_loader:
            inputs, scores, results = inputs.to(device), scores.to(device), results.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, scores, results)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, scores, results in val_loader:
                inputs, scores, results = inputs.to(device), scores.to(device), results.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, scores, results)
                val_loss += loss.item() * inputs.size(0)
                
        val_loss /= len(val_loader.dataset)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch:03d}/{args.epochs:03d}] -> LR hiện tại: {current_lr:.6f} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Lưu file gốc của PyTorch
            torch.save(model.state_dict(), args.out)
            print(f"    💾 [SAVE Pytorch] Đã cập nhật ma trận trọng số tối ưu tại epoch {epoch} vào '{args.out}'")
            
            # TỰ ĐỘNG XUẤT FILE .NPZ CHO ENGINE AI.PY DÙNG NGAY LẬP TỨC
            npz_path = args.out.replace('.pt', '.npz')
            export_to_numpy(model, npz_path, dynamic_scale_k)

        if epoch % args.ckpt_interval == 0 or epoch == args.epochs:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, checkpoint_path)
            print(f"    🛡️ [CHECKPOINT] Đã lưu bảo hiểm bộ nhớ tại epoch {epoch}")

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    print("\n=====================================================")
    print(f" 🏆 CHIẾN DỊCH HUẤN LUYỆN HOÀN TẤT! Best Val Loss đạt đỉnh: {best_val_loss:.6f}")
    print("=====================================================")

if __name__ == "__main__":
    train_nnue()