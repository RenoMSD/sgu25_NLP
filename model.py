import torch
import torch.nn as nn
import random 

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        # Bắt buộc dùng nn.LSTM theo yêu cầu đồ án
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_len):
        # src: [seq_len, batch_size]
        
        embedded = self.dropout(self.embedding(src))
        
        # ----------------------------------------------------
        # FIX CỦA LỖI RuntimeError: Đảm bảo src_len là list Python integers
        # ----------------------------------------------------
        # Chuyển đổi tensor src_len về CPU (nếu chưa) và sau đó về list Python
        lengths = src_len.cpu().tolist() 
        
        # Packing: Bắt buộc enforce_sorted=True
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths, enforce_sorted=True)
        
        # Encoder: Context Vector cố định
        packed_outputs, (hidden, cell) = self.rnn(packed_embedded)
        
        # hidden/cell: [n_layers, batch_size, hid_dim]
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg_input, hidden, cell):
        # trg_input: [batch_size] (1 token/step)
        
        trg_input = trg_input.unsqueeze(0) # [1, batch_size]
        
        embedded = self.dropout(self.embedding(trg_input))
        
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        
        prediction = self.fc_out(output.squeeze(0))
        
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trg, src_len, teacher_forcing_ratio = 0.5):
        
        trg_len, batch_size = trg.shape
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        # Context vector từ Encoder
        hidden, cell = self.encoder(src, src_len)
        
        # Token đầu tiên là <sos>
        trg_input = trg[0, :] 
        
        for t in range(1, trg_len):
            prediction, hidden, cell = self.decoder(trg_input, hidden, cell)
            
            outputs[t] = prediction
            
            top1 = prediction.argmax(1) 
            
            # Logic Teacher Forcing
            teacher_force = random.random() < teacher_forcing_ratio
            
            trg_input = trg[t, :] if teacher_force else top1
            
        return outputs