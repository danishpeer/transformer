import torch
from torch.utils.data import Dataset

class LangDataset(Dataset):
    def __init__(self, ds, src_tokenizer, out_tokenizer, src_lang, out_lang, seq_len) -> None:
        super().__init__()

        self.ds = ds
        self.src_tokenizer = src_tokenizer
        self.out_tokenizer = out_tokenizer
        self.src_lang = src_lang
        self.out_lang = out_lang
        self.seq_len = seq_len

        self.sos_token = torch.tensor([src_tokenizer.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = torch.tensor([src_tokenizer.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([src_tokenizer.token_to_id('[PAD]')], dtype=torch.int64)


    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        src_target_pair = self.ds[index]
        src_text  = src_target_pair['translation'][self.src_lang]
        out_text = src_target_pair['translation'][self.out_lang]

        src_tokens = self.src_tokenizer.encode(src_text).ids
        out_tokens = self.out_tokenizer.encode(out_text).ids

        src_num_pad = self.seq_len - len(src_tokens) - 2 
        out_num_pad = self.seq_len - len(out_tokens) - 1


        if src_num_pad < 0 or out_num_pad < 0:
            raise ValueError("Sentence is too long")

        encoder_tokens = torch.concat(
            [
                self.sos_token,
                torch.tensor(src_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]*src_num_pad, dtype=torch.int64)
            ]
        ) 

        decoder_tokens = torch.concat(
            [
                self.sos_token,
                torch.tensor(out_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token]*out_num_pad, dtype=torch.int64)
            ]
        ) 

        labels = torch.concat(
            [
                torch.tensor(out_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]*out_num_pad, dtype=torch.int64)
            ]
        ) 

        return {
            'encoder_input': encoder_tokens,
            'decoder_input': decoder_tokens,
            'encoder_mask': (encoder_tokens != self.pad_token).unsqueeze(0).unsqueeze(0).int(),  # [1,1,len] -> broadcasts to [h,len, len] in attention 
            'decoder_mask': (decoder_tokens != self.pad_token).unsqueeze(0).int() & causal_mask(self.seq_len), # [1,len, len]  # (1, len ) & (1,len, len)    
            'labels': labels,
            "src_text": src_text,
            "tgt_text": out_text,
        }
    

def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0

