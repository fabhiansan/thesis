import penman
from tqdm import tqdm
import torch
from transformers import T5TokenizerFast, AutoModelForSeq2SeqLM


class AMRToTextBase:
    def __call__(self, graphs: list[penman.Graph]) -> list[str]:
        """
        Transform all AMR graphs into sentences.

        Args:
        - `graphs`: List of AMR graph.
        """

        raise NotImplementedError("No implementation for base class.")
    
    
    
class AMRToTextWithTaufiqMethod(AMRToTextBase):
    """
    Class for transforming AMR to text, a.k.a. AMR generation, with Taufiq method
    ([code](https://github.com/taufiqhusada/amr-to-text-indonesia)).
    """

    def __init__(
            self,
            model_path: str,
            lowercase: bool = True,
            num_beams: int = 5,
            max_length: int = 384,
    ):
        """
        Initialize `AMRToTextWithTaufiqMethod` class.

        Args:
        - `model_path`: Model path. Make sure it contains model and tokenizer folders.`

        - `lowercase`: Is the model can only accept lowercase inputs?

        - `num_beams`: Number of beams used for generation.

        - `max_length`: Maximum length of prediction tokens.
        """

        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            print("Running on the GPU")
        else:
            device = torch.device("cpu")
            print("Running on the CPU")

        self.device = device

        self.tokenizer = T5TokenizerFast.from_pretrained(os.path.join(model_path, 'tokenizer'))

        model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(model_path, 'model'))
        for param in model.parameters():
            param.data = param.data.contiguous()

        #moving the model to device(GPU/CPU)
        model.to(device)
        model.eval()

        self.model = model
        self.lowercase = lowercase
        self.num_beams = num_beams
        self.max_length = max_length

    def __call__(self, graphs: list[penman.Graph]) -> list[str]:
        """
        Transform all AMR graphs into sentences.

        Args:
        - `graphs`: List of AMR graph.
        """
        sentences: list[str] = []

        for g in tqdm(graphs):
            no_metadata_g = make_no_metadata_graph(g)
            text = to_amr_with_pointer(
                penman.encode(no_metadata_g, indent=None)
            )

            if self.lowercase:
                text = text.lower()
            
            input_ids = self.tokenizer.encode(
                f"{T5_PREFIX}{text}",
                return_tensors="pt",
                add_special_tokens=False
            )  # Batch size 1

            input_ids = input_ids.to(self.device)
            outputs = self.model.generate(
                input_ids,
                num_beams=self.num_beams,
                max_length=self.max_length
            )

            gen_text: str = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            sentences.append(gen_text)
        
        return sentences
    
    
    