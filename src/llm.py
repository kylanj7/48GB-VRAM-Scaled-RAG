"""
HuggingFace LLM interface with quantization support
"""
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    pipeline
)
from config.settings import (
    LLM_MODEL,
    LLM_DEVICE,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
    DO_SAMPLE,
    USE_4BIT_QUANTIZATION,
    USE_8BIT_QUANTIZATION,
    TRUST_REMOTE_CODE,
    HF_CACHE_DIR
)


class HuggingFaceLLM:
    """Wrapper for HuggingFace causal language models with quantization"""
    
    def __init__(self):
        print(f"Loading LLM: {LLM_MODEL}")
        self.device = LLM_DEVICE if torch.cuda.is_available() else "cpu"
        
        # Configure quantization if enabled
        quantization_config = None
        if USE_4BIT_QUANTIZATION and self.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            print("  Using 4-bit quantization")
        elif USE_8BIT_QUANTIZATION and self.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
            print("  Using 8-bit quantization")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            LLM_MODEL,
            cache_dir=str(HF_CACHE_DIR),
            trust_remote_code=TRUST_REMOTE_CODE
        )
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL,
            quantization_config=quantization_config,
            device_map="auto" if self.device == "cuda" else None,
            cache_dir=str(HF_CACHE_DIR),
            trust_remote_code=TRUST_REMOTE_CODE,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        print(f"✓ LLM loaded on {self.device}")
    
    def generate(self, prompt: str, max_new_tokens: int = None) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Override default max tokens
            
        Returns:
            Generated text
        """
        tokens = max_new_tokens or MAX_NEW_TOKENS
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=tokens,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=DO_SAMPLE,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the generated tokens (skip input prompt)
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text.strip()
    
    def invoke(self, prompt: str) -> str:
        """Alias for generate() to match Ollama interface"""
        return self.generate(prompt)


def get_llm():
    """
    Initialize HuggingFace LLM.
    
    Returns:
        HuggingFaceLLM: Configured LLM
    """
    return HuggingFaceLLM()


if __name__ == "__main__":
    # Test LLM
    llm = get_llm()
    test_prompt = "Explain what a RAG system is in one sentence."
    
    try:
        result = llm.generate(test_prompt)
        print(f"\n✓ LLM working!\n\nPrompt: {test_prompt}\nResponse: {result}")
    except Exception as e:
        print(f"✗ LLM test failed: {e}")
