import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import (
    load_data, data_process, custom_collate_fn, DEVICE,
    vocab_src, vocab_trg, UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX
)
from model import Encoder, Decoder, Seq2Seq
import random
import time
import math
import numpy as np
import os
from nltk.translate.bleu_score import corpus_bleu # Dùng cho đánh giá cuối cùng

# ----------------------------------------------------
# KHỞI TẠO THƯ MỤC CẦN THIẾT
# ----------------------------------------------------
if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')

# ----------------------------------------------------
# THAM SỐ CẤU HÌNH (CONFIGURATIONS)
# ----------------------------------------------------
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True 

BATCH_SIZE = 128
N_EPOCHS = 20 
PATIENCE = 3 # Early Stopping: Dừng nếu val loss không giảm sau 3 epoch

# Tham số Mô hình (Khuyến nghị của Đồ án)
INPUT_DIM = len(vocab_src)
OUTPUT_DIM = len(vocab_trg)
EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
TEACHER_FORCING_RATIO = 0.5 # Yêu cầu bắt buộc 0.5
CLIP = 1.0 

print(f"Sử dụng thiết bị: {DEVICE}")
print(f"Kích thước Vocab (Source): {INPUT_DIM}")
print(f"Kích thước Vocab (Target): {OUTPUT_DIM}")


# ----------------------------------------------------
# KHỞI TẠO DỮ LIỆU VÀ MÔ HÌNH
# ----------------------------------------------------
try:
    train_pairs = load_data('data/raw/train.en', 'data/raw/train.fr')
    valid_pairs = load_data('data/raw/val.en', 'data/raw/val.fr')
    test_pairs = load_data('data/raw/test.en', 'data/raw/test.fr')

    train_data = data_process(train_pairs, vocab_src, vocab_trg)
    valid_data = data_process(valid_pairs, vocab_src, vocab_trg)
    test_data = data_process(test_pairs, vocab_src, vocab_trg)

    train_iterator = DataLoader(train_data, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn)
    valid_iterator = DataLoader(valid_data, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn)
    test_iterator = DataLoader(test_data, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn)
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file dữ liệu. Vui lòng kiểm tra thư mục 'data/raw'.")
    exit()

enc = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)

# Loss & Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001) 
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX) 

# ----------------------------------------------------
# HÀM HUẤN LUYỆN & ĐÁNH GIÁ (TRAIN & EVAL)
# ----------------------------------------------------

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, (src, trg, src_len) in enumerate(iterator):
        src, trg = src.to(DEVICE), trg.to(DEVICE)
        
        optimizer.zero_grad()
        
        # src_len được sử dụng trong Encoder (Packing)
        # TEACHER_FORCING_RATIO được sử dụng trong Seq2Seq
        output = model(src, trg, src_len, TEACHER_FORCING_RATIO)
        
        # Bỏ qua token SOS
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, (src, trg, src_len) in enumerate(iterator):
            src, trg = src.to(DEVICE), trg.to(DEVICE)

            # Tắt teacher forcing khi đánh giá (ratio = 0)
            output = model(src, trg, src_len, 0) 
            
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            
            loss = criterion(output, trg)
            epoch_loss += loss.item()
            
    return epoch_loss / len(iterator)

# ----------------------------------------------------
# VÒNG LẶP HUẤN LUYỆN CHÍNH (MAIN LOOP)
# ----------------------------------------------------

print("\n--- BẮT ĐẦU HUẤN LUYỆN ---")
best_valid_loss = float('inf')
patience_counter = 0

for epoch in range(N_EPOCHS):
    start_time = time.time()
    
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()
    epoch_mins = int((end_time - start_time) / 60)
    epoch_secs = int((end_time - start_time) % 60)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'checkpoints/best_model.pth')
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s | Val Loss IMPROVED. Saving model.')
    else:
        patience_counter += 1
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s | Val Loss did not improve. Patience: {patience_counter}/{PATIENCE}')
        
        # Early Stopping (Yêu cầu bắt buộc)
        if patience_counter >= PATIENCE:
            print(f"\n--- Early Stopping triggered after {epoch+1} epochs. ---")
            break

    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t  Val Loss: {valid_loss:.3f} |   Val PPL: {math.exp(valid_loss):7.3f}')

print("\n--- KẾT THÚC HUẤN LUYỆN ---")

# ----------------------------------------------------
# ĐÁNH GIÁ CUỐI CÙNG TRÊN TẬP TEST
# ----------------------------------------------------
# Tải lại mô hình tốt nhất
try:
    model.load_state_dict(torch.load('checkpoints/best_model.pth', map_location=DEVICE))
    test_loss = evaluate(model, test_iterator, criterion)
    print(f'ĐÁNH GIÁ CUỐI CÙNG:')
    print(f'Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f}')
    
    # Tiếp tục chạy translate.py để có BLEU Score và ví dụ dịch.
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file checkpoint. Đánh giá cuối cùng bị bỏ qua.")