from transformers import MBartForConditionalGeneration, AutoConfig
from model_interface.tokenization_bart import AMRBartTokenizer
import torch
import penman

class TextToAMRSan:
    def __init__(self, model_path="./models/mbart-en-id-smaller-indo-amr-parsing-translated-nafkhan"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load config
        self.config = AutoConfig.from_pretrained(model_path)
        
        # Initialize tokenizer with correct parameters
        self.tokenizer = AMRBartTokenizer.from_pretrained(
            model_path,
            use_fast=False
        )
        
        # Initialize model with config
        self.model = MBartForConditionalGeneration.from_pretrained(
            model_path,
            config=self.config
        ).to(self.device)
        
        # Set important parameters
        self.max_src_length = 400
        self.max_gen_length = 1024
        self.num_beams = 5
        
    def __call__(self, sentence: str) -> penman.Graph:
        # Prepare input with AMR special tokens
        raw_txt_ids = self.tokenizer(
            sentence,
            max_length=self.max_src_length,
            padding=False,
            truncation=True
        )["input_ids"]
        
        # Add AMR special tokens
        txt_ids = [raw_txt_ids[:self.max_src_length-3] + [
            self.tokenizer.amr_bos_token_id,
            self.tokenizer.mask_token_id,
            self.tokenizer.amr_eos_token_id
        ]]
        
        # Pad and convert to tensor
        txt_ids = self.tokenizer.pad(
            {"input_ids": txt_ids},
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate
        preds = self.model.generate(
            txt_ids["input_ids"],
            max_length=self.max_gen_length,
            num_beams=self.num_beams,
            use_cache=True,
            decoder_start_token_id=self.tokenizer.amr_bos_token_id,
            eos_token_id=self.tokenizer.amr_eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            no_repeat_ngram_size=0,
            min_length=0
        )
        
        # Decode AMR
        pred = preds[0]
        pred[0] = self.tokenizer.bos_token_id
        pred = [
            self.tokenizer.eos_token_id if itm == self.tokenizer.amr_eos_token_id else itm
            for itm in pred if itm != self.tokenizer.pad_token_id
        ]
        
        graph, status, (lin, backr) = self.tokenizer.decode_amr(
            pred,
            restore_name_ops=False
        )
        
        # Add metadata
        metadata = {
            "id": "0",
            "annotator": "TextToAMRSan",
            "snt": sentence
        }
        graph.metadata = metadata
        
        return graph