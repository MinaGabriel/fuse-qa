import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from prettytable import PrettyTable
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SRE:
    def __init__(self,
                 classifier_device="cuda:0",
                 base_model_name="google/gemma-2-2b-it",
                 lora_dir="MinaGabriel/gemma-2-2b-it-SRE-LoRA"):

        self.device = torch.device(classifier_device)
        self.base_model_name = base_model_name
        self.lora_dir = lora_dir

        self._load_models()

    def _load_models(self):
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # BASE MODEL (clean, no LoRA)
        logger.info("Loading BASE model...")
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_name,
            num_labels=2,
            torch_dtype=torch.float16
        ).to(self.device)

        # LoRA MODEL
        logger.info("Loading LoRA model...")
        base_for_lora = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_name,
            num_labels=2,
            torch_dtype=torch.float16
        )

        self.lora_model = PeftModel.from_pretrained(
            base_for_lora,
            self.lora_dir
        ).to(self.device)

        self.base_model.eval()
        self.lora_model.eval()

        #VERIFICATION
        logger.info("\n===== VERIFICATION =====")

        lora_layers = [n for n, _ in self.lora_model.named_modules() if "lora" in n.lower()]
        logger.info(f"LoRA layers found: {len(lora_layers)}")

        if len(lora_layers) == 0:
            logger.error("LoRA NOT loaded correctly")

        logger.info("===== READY =====\n")

    def build_text(self, question, full_context, sentence):
        return (
            f"Question: {question}\n\n"
            f"Context:\n{full_context}\n\n"
            f"Sentence:\n{sentence}\n\n"
            f"Is this sentence useful? Answer Yes or No.\nAnswer:"
        )

    @torch.no_grad()
    def compare_base_vs_lora(self, question, full_context, sentences: List[str]):

        texts = [self.build_text(question, full_context, s) for s in sentences]

        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(self.device)

        # BASE
        base_logits = self.base_model(**enc).logits
        base_probs = F.softmax(base_logits, dim=-1)

        # LoRA
        lora_logits = self.lora_model(**enc).logits
        lora_probs = F.softmax(lora_logits, dim=-1)

        results = []
        for i, s in enumerate(sentences):
            results.append({
                "sentence": s,
                "base_yes": base_probs[i, 1].item(),
                "lora_yes": lora_probs[i, 1].item(),
                "delta": (lora_probs[i, 1] - base_probs[i, 1]).item()
            })

        return sorted(results, key=lambda r: r["lora_yes"], reverse=True)

    def print_table(self, results):
        table = PrettyTable()
        table.field_names = ["Rank", "Base Yes", "LoRA Yes", "Δ Impact", "Sentence"]

        for i, r in enumerate(results, 1):
            table.add_row([
                i,
                f"{r['base_yes']:.4f}",
                f"{r['lora_yes']:.4f}",
                f"{r['delta']:+.4f}",
                r["sentence"]
            ])

        print(table)
        
    @torch.no_grad()
    def score_sentences_batch(
        self,
        question,
        full_context,
        sentences,
        max_length=1024,
        batch_size=16
    ):
        all_results = []

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]

            texts = [
                self.build_text(question, full_context, s)
                for s in batch
            ]

            enc = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(self.device)

            # Always use LoRA model for SRE scoring
            logits = self.lora_model(**enc).logits
            probs = F.softmax(logits, dim=-1)

            for j, s in enumerate(batch):
                all_results.append({
                    "sentence": s,
                    "score": probs[j, 1].item(),   # YES
                    "no_score": probs[j, 0].item()
                })

        return sorted(all_results, key=lambda x: x["score"], reverse=True)
    
    def print_ranking_table(self, results, top_k=10):
        from prettytable import PrettyTable

        table = PrettyTable()
        table.field_names = ["Rank", "Score", "Sentence"]

        for i, r in enumerate(results[:top_k], 1):
            table.add_row([
                i,
                f"{r['score']:.4f}",
                r["sentence"]
            ])

        print(table)
