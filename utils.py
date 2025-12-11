import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import spacy
import numpy as np 
from collections import Counter 
import os
# KHÔNG CẦN IMPORT GÌ TỪ TORCHTEXT.VOCAB NỮA!

# ----------------------------------------------------
# 1. KHỞI TẠO SPACY VÀ HÀM TOKENIZE
# ----------------------------------------------------

# Tải Spacy model
try:
    spacy_en = spacy.load("en_core_web_sm")
    spacy_fr = spacy.load("fr_core_news_sm")
except OSError:
    print("Lỗi: Không tìm thấy Spacy model. Vui lòng chạy 'python -m spacy download en_core_web_sm' và 'fr_core_news_sm'")
    exit()

def tokenize_en(text):
    """Tokenize tiếng Anh"""
    return [tok.text.lower() for tok in spacy_en.tokenizer(text)]

def tokenize_fr(text):
    """Tokenize tiếng Pháp"""
    return [tok.text.lower() for tok in spacy_fr.tokenizer(text)]

def load_data(filepath_en, filepath_fr):
    """Đọc dữ liệu từ file và trả về danh sách các cặp câu (en, fr)"""
    with open(filepath_en, encoding='utf-8') as f_en, open(filepath_fr, encoding='utf-8') as f_fr:
        raw_pairs = []
        for en_line, fr_line in zip(f_en, f_fr):
            raw_pairs.append((en_line.strip(), fr_line.strip()))
    return raw_pairs

# Tải dữ liệu huấn luyện (dùng để xây dựng vocab)
try:
    train_pairs = load_data('data/raw/train.en', 'data/raw/train.fr')
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file dữ liệu 'data/raw/train.en' hoặc 'train.fr'.")
    print("Vui lòng đảm bảo các file đã được đặt đúng chỗ trước khi chạy.")
    train_pairs = []
    
# Token đặc biệt và giới hạn kích thước Vocab
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
SPECIAL_TOKENS = ['<unk>', '<pad>', '<sos>', '<eos>']
VOCAB_SIZE_LIMIT = 10000 # Yêu cầu giới hạn 10.000 từ


# ----------------------------------------------------
# 2. CUSTOM VOCAB CLASS (XÂY DỰNG LẠI CHO ỔN ĐỊNH)
# ----------------------------------------------------

class CustomVocab:
    """Lớp Vocab tùy chỉnh để thay thế torchtext.vocab"""
    def __init__(self, token_list, specials, default_index):
        # Xây dựng stoi (String to Index) dictionary
        # Bắt đầu đánh index từ 0 cho SPECIAL_TOKENS
        self.stoi = {token: i for i, token in enumerate(specials)}
        current_index = len(specials)
        
        # Thêm các token phổ biến (từ list 10k)
        for token in token_list:
            if token not in self.stoi:
                self.stoi[token] = current_index
                current_index += 1
                
        # Xây dựng itos (Index to String) list
        self.itos = specials + [token for token in token_list if token not in specials]
        self.default_index = default_index

    def get_stoi(self):
        """Trả về dictionary String to Index"""
        return self.stoi

    def get_itos(self):
        """Trả về list Index to String"""
        return self.itos
    
    def __len__(self):
        return len(self.itos)

    def set_default_index(self, index):
        # Giữ lại để tương thích với các bước sau
        pass

# Hàm hỗ trợ tính tần suất
def get_token_frequency(data_pairs, tokenizer, lang_index):
    """Tính tần suất của tất cả token trong tập dữ liệu."""
    counter = Counter()
    for pair in data_pairs:
        text = pair[lang_index] # 0 cho English, 1 cho French
        counter.update(tokenizer(text))
    return counter

# 1. Tính tần suất
src_counter = get_token_frequency(train_pairs, tokenize_en, 0)
trg_counter = get_token_frequency(train_pairs, tokenize_fr, 1)

# 2. Lấy Top 10.000 từ phổ biến nhất (Yêu cầu bắt buộc)
# most_common(N) trả về list các cặp (token, count)
src_vocab_list = [item[0] for item in src_counter.most_common(VOCAB_SIZE_LIMIT)]
trg_vocab_list = [item[0] for item in trg_counter.most_common(VOCAB_SIZE_LIMIT)]

# 3. Xây dựng Vocab object bằng Custom Class
vocab_src = CustomVocab(src_vocab_list, SPECIAL_TOKENS, UNK_IDX)
vocab_trg = CustomVocab(trg_vocab_list, SPECIAL_TOKENS, UNK_IDX)

# Đặt index cho <unk> (đã được xử lý trong constructor)
vocab_src.set_default_index(UNK_IDX)
vocab_trg.set_default_index(UNK_IDX)


# ----------------------------------------------------
# 3. DATASET VÀ DATALOADER
# ----------------------------------------------------

def data_process(text_pairs, vocab_src, vocab_trg):
    """Chuyển đổi các cặp câu (text) sang cặp tensor."""
    data = []
    # Lấy hàm get_stoi() để tra cứu index của token
    src_stoi = vocab_src.get_stoi()
    trg_stoi = vocab_trg.get_stoi()
    
    for en_text, fr_text in text_pairs:
        # Source (tiếng Anh)
        # Sử dụng .get(token, UNK_IDX) để xử lý OOV
        src_tensor = [src_stoi.get(token, UNK_IDX) for token in tokenize_en(en_text)]
        # Target (tiếng Pháp): Thêm SOS và EOS
        trg_tensor = [SOS_IDX] + [trg_stoi.get(token, UNK_IDX) for token in tokenize_fr(fr_text)] + [EOS_IDX]
        
        if src_tensor and trg_tensor:
            data.append((torch.tensor(src_tensor, dtype=torch.long), torch.tensor(trg_tensor, dtype=torch.long)))
    return data


class TranslationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def custom_collate_fn(batch):
    """
    Collate function tùy chỉnh để:
    1. Sắp xếp batch theo độ dài giảm dần của câu nguồn (SRC) - Bắt buộc cho Packing.
    2. Thực hiện Padding.
    """
    # 1. Tách và Sắp xếp
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    
    src_tensors = [item[0] for item in batch]
    trg_tensors = [item[1] for item in batch]
    
    # 2. Padding
    # pad_sequence (batch_first=False) -> [seq_len, batch_size]
    src_padded = pad_sequence(src_tensors, padding_value=PAD_IDX, batch_first=False)
    trg_padded = pad_sequence(trg_tensors, padding_value=PAD_IDX, batch_first=False)
    
    # 3. Lấy độ dài thực
    src_lengths = torch.tensor([len(tensor) for tensor in src_tensors], dtype=torch.long)
    
    return src_padded, trg_padded, src_lengths

# Device (Định nghĩa ở đây để dễ dàng import sang file khác)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')