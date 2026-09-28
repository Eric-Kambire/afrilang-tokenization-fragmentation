import argparse, json, math, random, os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import sentencepiece as spm

class TokenDataset(Dataset):
    def __init__(self, ids, seq_len):
        self.ids = ids
        self.seq_len = seq_len
    def __len__(self):
        return max(0, (len(self.ids)-1)//self.seq_len)
    def __getitem__(self, i):
        s = i*self.seq_len
        x = torch.tensor(self.ids[s:s+self.seq_len], dtype=torch.long)
        y = torch.tensor(self.ids[s+1:s+self.seq_len+1], dtype=torch.long)
        return x,y

class TinyTransformer(nn.Module):
    def __init__(self, vocab, d_model=256, n_heads=4, n_layers=4, d_ff=1024, max_len=128, dropout=0.1):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
                                           dropout=dropout, batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.tok.weight
        self.max_len = max_len
    def forward(self, x):
        b,t = x.shape
        p = torch.arange(t, device=x.device)
        h = self.tok(x) + self.pos(p)[None,:,:]
        mask = torch.full((t,t), float("-inf"), device=x.device)
        mask = torch.triu(mask, diagonal=1)
        h = self.enc(h, mask=mask)
        return self.head(self.ln(h))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--spm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--d_ff", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--resume", action="store_true")
    args=ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sp = spm.SentencePieceProcessor(model_file=args.spm)
    text = Path(args.text).read_text(encoding="utf-8")
    ids = sp.encode(text, out_type=int)
    ds = TokenDataset(ids, args.seq_len)
    if len(ds) == 0: raise ValueError("Corpus too small for seq_len")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    model = TinyTransformer(sp.get_piece_size(), args.d_model, args.n_heads, args.n_layers, args.d_ff, args.seq_len).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(device=="cuda"))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    ckpt = out.with_suffix(".pt")
    start=0
    if args.resume and ckpt.exists():
        obj=torch.load(ckpt, map_location=device)
        model.load_state_dict(obj["model"]); opt.load_state_dict(obj["opt"]); start=obj["step"]+1

    loss_fn=nn.CrossEntropyLoss()
    it=iter(dl)
    model.train()
    running=[]
    for step in range(start,args.steps):
        opt.zero_grad(set_to_none=True)
        total=0.0
        for _ in range(args.grad_accum):
            try: x,y=next(it)
            except StopIteration:
                it=iter(dl); x,y=next(it)
            x,y=x.to(device),y.to(device)
            with torch.amp.autocast("cuda", enabled=(device=="cuda")):
                logits=model(x)
                loss=loss_fn(logits.reshape(-1,logits.size(-1)),y.reshape(-1))/args.grad_accum
            scaler.scale(loss).backward()
            total += loss.item()
        scaler.step(opt); scaler.update()
        running.append(total)
        if step % 20 == 0:
            print(f"step={step} loss={total:.4f} device={device}")
        if step % 100 == 0 or step == args.steps-1:
            torch.save({"model":model.state_dict(),"opt":opt.state_dict(),"step":step,"args":vars(args)}, ckpt)
    result={"device":device,"steps":args.steps,"final_loss":float(sum(running[-20:])/max(1,len(running[-20:]))),
            "tokens":len(ids),"vocab":sp.get_piece_size()}
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__=="__main__":
    main()
