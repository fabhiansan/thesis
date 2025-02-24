import torch
from utils.data_utils import T5_PREFIX

def generate(text, model, tokenizer, num_beams, model_type, device='cpu'):
    if (model_type=='indo-bart'):
        return _generate_bart(text, model, tokenizer, num_beams, device)
    elif (model_type=='indo-t5'):
        return _generate_t5(text, model, tokenizer, num_beams, device)
    else:
        raise ValueError(f'Unknown model_type `{model_type}`')

def _generate_bart(text, model, tokenizer, num_beams, device):
    text = text.lower()
    model.to(device)

    model.eval()
    bart_input = tokenizer.prepare_input_for_generation(text,
                                         lang_token = '[indonesian]', decoder_lang_token='[indonesian]')
    input_ids = torch.tensor([bart_input['input_ids']]).to(device)
    print(input_ids)
    outputs = model.generate(input_ids, num_beams=num_beams)
    
    gen_text= tokenizer.decode(outputs[0], skip_special_tokens=True)
    return gen_text

def _generate_t5(text, model, tokenizer, num_beams, device):
    text = text.lower()
    model.to(device)
    
    model.eval()
    input_ids = tokenizer.encode(f"{T5_PREFIX}{text}", return_tensors="pt", add_special_tokens=False)  # Batch size 1
    input_ids = input_ids.to(device)
    outputs = model.generate(input_ids, num_beams=num_beams)
    
    gen_text= tokenizer.decode(outputs[0], skip_special_tokens=True)
    return gen_text