import streamlit as st
import torch
import torch.nn as nn
import json
import re
from torch.utils.data import Dataset, DataLoader
import math
import zipfile

with zipfile.ZipFile("best_model.zip", "r") as zip_ref:
    zip_ref.extractall(".")  # Extracts to current directory
    
# Define model components (same as your original code, but condensed)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        attention_probs = torch.softmax(attention_scores, dim=-1)
        return torch.matmul(attention_probs, V)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        Q = self.W_q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(output)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        return self.norm2(x + self.dropout(ff_output))

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.cross_attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        attn_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attention(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        return self.norm3(x + self.dropout(ff_output))

class TransformerSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_heads=8, n_layers=6, d_ff=2048, dropout=0.1, max_len=5000):
        super().__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src_emb = self.dropout(self.positional_encoding(self.encoder_embedding(src) * (self.d_model ** 0.5)))
        tgt_emb = self.dropout(self.positional_encoding(self.decoder_embedding(tgt) * (self.d_model ** 0.5)))
        enc_output = src_emb
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
        dec_output = tgt_emb
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
        return self.fc_out(dec_output)

# Load vocabulary safely
def load_vocab(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Vocabulary file {file_path} not found.")
        return {}

# Improved tokenization function
def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text)

# Convert text to token IDs
def text_to_ids(text, vocab):
    return [vocab.get("<SOS>", 1)] + [vocab.get(token, vocab.get("<UNK>", 0)) for token in tokenize(text)] + [vocab.get("<EOS>", 2)]

# Generate C++ code safely
def generate_cpp(model, pseudo_input, pseudo_vocab, cpp_vocab, cpp_inv_vocab, device, max_len=50):
    model.eval()
    with torch.no_grad():
        src_ids = text_to_ids(pseudo_input, pseudo_vocab)
        src = torch.tensor([src_ids], dtype=torch.long).to(device)
        tgt_input = torch.tensor([[cpp_vocab.get("<SOS>", 1)]], dtype=torch.long).to(device)
        
        for _ in range(max_len):
            tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
            output = model(src, tgt_input, tgt_mask=tgt_mask)
            next_token = output.argmax(dim=-1)[:, -1].unsqueeze(1)
            tgt_input = torch.cat([tgt_input, next_token], dim=1)
            if next_token.item() == cpp_vocab.get("<EOS>", 2):
                break
        
        pred_tokens = [cpp_inv_vocab.get(idx.item(), "<UNK>") for idx in tgt_input[0][1:] if idx.item() != cpp_vocab.get("<PAD>", 0)]
        return " ".join(pred_tokens)

# Streamlit app
def main():
    st.title("Pseudocode to C++ Converter")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with zipfile.ZipFile("best_model.zip", "r") as zip_ref:
        zip_ref.extractall(".")
    pseudo_vocab = load_vocab("pseudo_vocab.json")
    cpp_vocab = load_vocab("cpp_vocab.json")
    cpp_inv_vocab = load_vocab("cpp_inv_vocab.json")
    
    if not pseudo_vocab or not cpp_vocab or not cpp_inv_vocab:
        st.error("Error loading vocabularies. Check the files.")
        return
    
    # Load model
    model = TransformerSeq2Seq(
        src_vocab_size=len(pseudo_vocab),
        tgt_vocab_size=len(cpp_vocab),
        d_model=512, n_heads=8, n_layers=6, d_ff=2048, dropout=0.2
    ).to(device)
    try:
        model.load_state_dict(torch.load("best_model.pth", map_location=device))
        model.eval()
    except FileNotFoundError:
        st.error("Model file not found.")
        return
    
    option = st.radio("Choose input method:", ("Text Input", "File Upload"))
    
    if option == "Text Input":
        pseudo_input = st.text_area("Enter Pseudocode:", height=200)
        if st.button("Convert") and pseudo_input:
            cpp_output = generate_cpp(model, pseudo_input, pseudo_vocab, cpp_vocab, cpp_inv_vocab, device)
            st.subheader("Generated C++ Code:")
            st.code(cpp_output, language="cpp")
    
    elif option == "File Upload":
        uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
        if uploaded_file and st.button("Convert"):
            pseudo_input = uploaded_file.read().decode("utf-8")
            cpp_output = generate_cpp(model, pseudo_input, pseudo_vocab, cpp_vocab, cpp_inv_vocab, device)
            st.subheader("Generated C++ Code:")
            st.code(cpp_output, language="cpp")

if __name__ == "__main__":
    main()