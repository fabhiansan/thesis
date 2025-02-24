import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

T5_PREFIX = "translate graph to indonesian: "

# class to load preprocessed amr data
class AMRToTextDataset(Dataset):    
    def __init__(self, file_amr_path, file_sent_path, tokenizer, split, file_level_path = None):
        temp_list_amr_input = []
        with open(file_amr_path, encoding='utf8') as f:
            temp_list_amr_input = f.readlines()
        list_amr_input = []
        for item in temp_list_amr_input:
            list_amr_input.append(item.strip().lower())  # lowercase for bart tokenizer
            
        temp_list_sent_output = []
        with open(file_sent_path, encoding='utf8') as f:
            temp_list_sent_output = f.readlines()
        list_sent_output = []
        for item in temp_list_sent_output:
            list_sent_output.append(item.strip().lower())
        
        if (file_level_path):
            temp_list_level = []
            with open(file_level_path) as f:
                temp_list_level = f.readlines()
            list_level = []
            for item in temp_list_level:
                list_level.append(item.strip())
            
            df = pd.DataFrame(list(zip(list_amr_input, list_sent_output, list_level)), columns = ['amr','sent','level'])
            self.with_tree_level = True
        else:
            df = pd.DataFrame(list(zip(list_amr_input, list_sent_output)), columns = ['amr','sent'])
            self.with_tree_level = False
        self.data = df
        self.tokenizer = tokenizer
 
    
    def __getitem__(self, index):
        data = self.data.loc[index,:]
        amr, sent = data['amr'], data['sent']
       
        tokenize_amr = self.tokenizer.encode(amr, add_special_tokens=False)
        tokenize_sent = self.tokenizer.encode(sent, add_special_tokens=False)
        
        item = {'input':{}, 'output':{}}
        item['input']['encoded'] = tokenize_amr
        item['input']['raw'] = amr
        item['output']['encoded'] = tokenize_sent
        item['output']['raw'] = sent
        
        if (self.with_tree_level):
            level = data['level']
            tokenize_amr, tokenize_level = self._encode_tokens_with_tree_level(amr, level)
            item['input']['encoded'] = tokenize_amr
            item['input']['raw'] = amr
            
            item['level'] = {}
            item['level']['encoded'] = tokenize_level
            item['level']['raw'] = level
            
        return item
    
    def _encode_tokens_with_tree_level(self, amr, level):
        list_level = [int(e) for e in level.split()]    
        token_ids = []
        tree_ids = []
        for i, token in enumerate(amr.split()):
            encoded_token = self.tokenizer.encode(token, add_special_tokens=False)
            token_ids += encoded_token

            # level start at 3, following https://github.com/EagleW/Stage-wise-Fine-tuning/blob/0286726e28f517793216a6c113b4a6a8d21fd906/preprocess/webnlg_tree.py#L101
            modified_level = list_level[i] + 2 if list_level[i]!= 0 else list_level[i]
            tree_ids += [modified_level] * len(encoded_token)
                
        tokenize_amr = token_ids
        tokenize_level = tree_ids
        return tokenize_amr, tokenize_level
    
    def __len__(self):
        return len(self.data)
        
## Data loader class
## This dataloader class for T5 and BART is using code (but modified a little bit) from https://github.com/indobenchmark/indonlg/blob/master/utils/data_utils.py
class AMRToTextDataLoader(DataLoader):
    def __init__(self, max_seq_len_amr=512, max_seq_len_sent=384, label_pad_token_id=-100, model_type='indo-t5', tokenizer=None, with_tree_level=False, *args, **kwargs):
        super(AMRToTextDataLoader, self).__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.max_seq_len_amr = max_seq_len_amr
        self.max_seq_len_sent = max_seq_len_sent
        
        self.pad_token_id = tokenizer.pad_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        
        self.label_pad_token_id = label_pad_token_id
        self.with_tree_level = with_tree_level
        
        if model_type == 'indo-t5' or model_type == 'mT5':
            self.bos_token_id = tokenizer.pad_token_id
            self.t5_prefix =np.array(self.tokenizer.encode(T5_PREFIX, add_special_tokens=False))
            self.collate_fn = self._t5_collate_fn
        elif model_type == 'indo-bart':
            source_lang = "[indonesian]"
            target_lang = "[indonesian]"

            self.src_lid_token_id = tokenizer.special_tokens_to_ids[source_lang]
            self.tgt_lid_token_id = tokenizer.special_tokens_to_ids[target_lang]
            self.collate_fn = self._bart_collate_fn
        else:
            raise ValueError(f'Unknown model_type `{model_type}`')
            
    def _t5_collate_fn(self, batch):
        batch_size = len(batch)
        if (self.with_tree_level):
            max_enc_len = min(self.max_seq_len_amr, max(map(lambda x: len(x['input']['encoded'])+1, batch))  + len(self.t5_prefix))
        else:
            max_enc_len = min(self.max_seq_len_amr, max(map(lambda x: len(x['input']['encoded']), batch))  + len(self.t5_prefix))
        max_dec_len = min(self.max_seq_len_sent, max(map(lambda x: len(x['output']['encoded']), batch)) + 1)
        
        
        enc_batch = np.full((batch_size, max_enc_len), self.pad_token_id, dtype=np.int64)
        dec_batch = np.full((batch_size, max_dec_len), self.pad_token_id, dtype=np.int64)
        label_batch = np.full((batch_size, max_dec_len), self.label_pad_token_id, dtype=np.int64)
        enc_mask_batch = np.full((batch_size, max_enc_len), 0, dtype=np.float32)
        dec_mask_batch = np.full((batch_size, max_dec_len), 0, dtype=np.float32)
        level_batch = None
        
        if (self.with_tree_level):
            level_batch = np.full((batch_size, max_enc_len), self.pad_token_id, dtype=np.int64)
        
        for i, item in enumerate(batch):
            input_seq = item['input']['encoded']
            if (self.with_tree_level):
                input_seq.append(self.eos_token_id)
            label_seq = item['output']['encoded']
            input_seq, label_seq = input_seq[:max_enc_len - len(self.t5_prefix)], label_seq[:max_dec_len - 1]
            
            # Assign content
            enc_batch[i,len(self.t5_prefix):len(self.t5_prefix) + len(input_seq)] = input_seq
            dec_batch[i,1:1+len(label_seq)] = label_seq
            label_batch[i,:len(label_seq)] = label_seq
            enc_mask_batch[i,:len(input_seq) + len(self.t5_prefix)] = 1
            dec_mask_batch[i,:len(label_seq) + 1] = 1
            
            # Assign special token to encoder input
            enc_batch[i,:len(self.t5_prefix)] = self.t5_prefix
            
            # Assign special token to decoder input
            dec_batch[i,0] = self.bos_token_id
            
            # Assign special token to label
            label_batch[i,len(label_seq)] = self.eos_token_id
            
            if (self.with_tree_level):
                level_seq = item['level']['encoded']
                level_seq.append(self.eos_token_id)
                level_seq = level_seq[:max_enc_len - len(self.t5_prefix)]  # truncate if greater than max len
                
                level_batch[i,len(self.t5_prefix):len(self.t5_prefix) + len(level_seq)] = level_seq  # assign content
                level_batch[i,:len(self.t5_prefix)] = [0]*len(self.t5_prefix)
            
        
        return enc_batch, dec_batch, enc_mask_batch, None, label_batch, level_batch
    
    def _bart_collate_fn(self, batch):
        # encoder input
        # <sent><eos><langid>
        # decoder input - 
        # <langid><sent><eos>
        # decoder output
        # <sent><eos><langid>
        
        batch_size = len(batch)
        max_enc_len = min(self.max_seq_len_amr, max(map(lambda x: len(x['input']['encoded']), batch)) + 2) # + 2 for eos and langid
        max_dec_len = min(self.max_seq_len_sent, max(map(lambda x: len(x['output']['encoded']), batch)) + 2) # + 2 for eos and langid
        
        enc_batch = np.full((batch_size, max_enc_len), self.pad_token_id, dtype=np.int64)
        dec_batch = np.full((batch_size, max_dec_len), self.pad_token_id, dtype=np.int64)
        label_batch = np.full((batch_size, max_dec_len), self.label_pad_token_id, dtype=np.int64)
        enc_mask_batch = np.full((batch_size, max_enc_len), 0, dtype=np.float32)
        dec_mask_batch = np.full((batch_size, max_dec_len), 0, dtype=np.float32)
        level_batch = None
        
        if (self.with_tree_level):
            level_batch = np.full((batch_size, max_enc_len), self.pad_token_id, dtype=np.int64)

        for i, item in enumerate(batch):
            input_seq = item['input']['encoded']
            label_seq = item['output']['encoded']
            input_seq, label_seq = input_seq[:max_enc_len-2], label_seq[:max_dec_len - 2]
            
            # Assign content
            enc_batch[i,0:len(input_seq)] = input_seq
            dec_batch[i,1:1+len(label_seq)] = label_seq
            label_batch[i,:len(label_seq)] = label_seq
            enc_mask_batch[i,:len(input_seq) + 2] = 1
            dec_mask_batch[i,:len(label_seq) + 2] = 1
            
            # Assign special token to encoder input
            enc_batch[i,len(input_seq)] = self.eos_token_id
            enc_batch[i,1+len(input_seq)] = self.src_lid_token_id
            
            # Assign special token to decoder input
            dec_batch[i,0] = self.tgt_lid_token_id
            dec_batch[i,1+len(label_seq)] = self.eos_token_id
            
            # Assign special token to label
            label_batch[i,len(label_seq)] = self.eos_token_id
            label_batch[i,1+len(label_seq)] = self.tgt_lid_token_id

            if (self.with_tree_level):
                level_seq = item['level']['encoded']

                for idx_level in range(len(level_seq)):
                    if (level_seq[idx_level]==0):
                        level_seq[idx_level] = self.pad_token_id

                level_seq = level_seq[:max_enc_len-2]
                level_batch[i,0:len(level_seq)] = level_seq
                enc_batch[i,len(input_seq)] = self.eos_token_id
                enc_batch[i,1+len(input_seq)] = self.pad_token_id
        
        return enc_batch, dec_batch, enc_mask_batch, None, label_batch, level_batch