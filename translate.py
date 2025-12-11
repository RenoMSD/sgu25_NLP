import torch
from utils import (
    tokenize_en, vocab_src, vocab_trg, 
    UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX, DEVICE
)
from model import Encoder, Decoder, Seq2Seq
from nltk.translate.bleu_score import corpus_bleu
from utils import load_data, data_process, custom_collate_fn
from torch.utils.data import DataLoader

# Tải mô hình tốt nhất và cấu hình
INPUT_DIM = len(vocab_src)
OUTPUT_DIM = len(vocab_trg)
EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)
try:
    model.load_state_dict(torch.load('checkpoints/best_model.pth', map_location=DEVICE))
    model.eval()
    print("Loaded best model checkpoint.")
except FileNotFoundError:
    print("Error: best_model.pth not found. Run train.py first.")

def translate(sentence, model, vocab_src, vocab_trg, max_len=50):
    """
    Hàm dịch một câu tiếng Anh -> tiếng Pháp (Yêu cầu bắt buộc)
    Sử dụng Greedy Decoding (Yêu cầu bắt buộc)
    """
    model.eval()
    
    # 1. Tokenize -> tensor
    tokens = [token for token in tokenize_en(sentence)]
    # Chuyển đổi sang index. Dùng .get(token, UNK_IDX) để xử lý OOV
    indexes = [vocab_src.get_stoi().get(token, UNK_IDX) for token in tokens]
    
    src_tensor = torch.LongTensor(indexes).to(DEVICE)
    # src_tensor: [src_len]
    
    # Thêm batch size dimension (batch_first=False -> [src_len, 1])
    src_tensor = src_tensor.unsqueeze(1)
    src_len = torch.LongTensor([len(indexes)])
    
    # 2. Encoder
    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor, src_len)
    
    # 3. Decoder
    # Token đầu tiên là <sos>
    trg_input = torch.LongTensor([SOS_IDX]).to(DEVICE) # [1]
    
    translated_tokens = []
    
    for t in range(max_len): # max_len=50 (Yêu cầu bắt buộc)
        
        with torch.no_grad():
            # trg_input: [1] -> [1, 1] (unsqueeze(1) trong Decoder)
            output, hidden, cell = model.decoder(trg_input, hidden, cell)
        
        # Greedy Decoding: Chọn token có xác suất cao nhất (Yêu cầu bắt buộc)
        pred_token = output.argmax(1).item()
        
        # Dừng khi gặp <eos>
        if pred_token == EOS_IDX:
            break
            
        translated_tokens.append(vocab_trg.get_itos()[pred_token])
        
        # Input cho bước tiếp theo là token dự đoán
        trg_input = torch.LongTensor([pred_token]).to(DEVICE)
        
    # 4. Detokenize
    return " ".join(translated_tokens)

def calculate_bleu(model, test_pairs, vocab_src, vocab_trg):
    """Tính BLEU score trung bình trên tập test."""
    test_iterator = DataLoader(
        data_process(test_pairs, vocab_src, vocab_trg), 
        batch_size=1, 
        collate_fn=custom_collate_fn
    )
    
    # Chuẩn bị dữ liệu cho corpus_bleu
    # targets: [[['ref1', 'ref2', ...]], [['ref1', 'ref2', ...]], ...]
    # predictions: ['hyp1', 'hyp2', ...]
    
    all_references = []
    all_translations = []

    for src_tensor, trg_tensor, src_len in test_iterator:
        # Lấy câu tiếng Anh gốc từ tensor
        src_tokens = [vocab_src.get_itos()[i] for i in src_tensor.squeeze(1).tolist() if i != PAD_IDX]
        src_sentence = " ".join(src_tokens)

        # Dịch câu
        predicted_sentence = translate(src_sentence, model, vocab_src, vocab_trg)
        all_translations.append(predicted_sentence.split())
        
        # Lấy câu tiếng Pháp gốc làm reference
        trg_tokens = [vocab_trg.get_itos()[i] for i in trg_tensor.squeeze(1).tolist()]
        # Loại bỏ <sos>, <eos>, <pad> khỏi reference
        reference = [tok for tok in trg_tokens if tok not in ['<sos>', '<eos>', '<pad>']]
        
        # NLTK cần list of list of references (dù chỉ có 1 reference)
        all_references.append([reference])

    # Tính toán BLEU score (Corpus BLEU)
    bleu = corpus_bleu(all_references, all_translations)
    return bleu

# --- Ví dụ sử dụng ---
if __name__ == '__main__':
    # Tải dữ liệu test một lần nữa để tính BLEU
    test_pairs_raw = load_data('data/raw/test.en', 'data/raw/test.fr')
    
    # Tính BLEU score
    bleu_score = calculate_bleu(model, test_pairs_raw, vocab_src, vocab_trg)
    print(f"\nBLEU Score trên tập Test: {bleu_score*100:.2f}%")
    
    # Dịch 5 ví dụ (Yêu cầu phân tích lỗi)
    examples = [
        "A girl is walking in a field.",
        "The black dog jumped over the fence.",
        "I need help with my homework.",
        "The train leaves at 6 o'clock.",
        "He always eats a sandwich for lunch."
    ]
    
    print("\n--- 5 VÍ DỤ DỊCH MẪU ---")
    for i, sen in enumerate(examples):
        translation = translate(sen, model, vocab_src, vocab_trg)
        print(f"{i+1}. ENG: {sen}")
        print(f"   FRA: {translation}")