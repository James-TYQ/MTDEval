from dataclasses import dataclass, field
from typing import Optional, List
from transformers import TrainingArguments as BaseTrainingArguments

@dataclass
class ScriptArguments:
    text_field: List[str] = field(
        default_factory=lambda: ["instruction", "dialogue_a", "dialogue_b"],
        metadata={
            "help": "Name of the text field in the dataset."
        },
    )       
    label_field: List[str] = field(
        default_factory=lambda: ['Accuracy', 'Logicality', 'Fluency', 'Relevance', 'Personalization', 'Creativity', 'Interactivity', 'Emotionality', 'Knowledge', 'Safety'],
        metadata={
            "help": "Name of the label field in the dataset."
        },
    )
    dimension_weights: List[float] = field(
        default_factory=lambda: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  
        metadata={
            "help": "Weights for each dimension in loss calculation."
        },
    )
    train_datasets_dir: str = field(
        default=None,     
        metadata={"help": "Directory of training datasets."},
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    max_length: Optional[int] = field(
        default=2048,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )

    lora: bool = field(default=False, metadata={"help": "Whether to use parameter efficient fine-tuning."})
    lora_path: str = field(default=None, metadata={"help": "Path to the lora model."})
    lora_modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    lora_r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    lora_target_modules: str = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    lora_alpha: int = field(default=16, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "Lora dropout"})

    single_label_ablation: int = field(default=-1, metadata={"help": "Whether to use single label for ablation."})

@dataclass
class TrainingArguments(BaseTrainingArguments):
    label_temperature: float = field(
        default=2.0,
        metadata={"help": "Label temperature"},
    )
    log_confidences: List[float] = field(
        default_factory=lambda: [0.5, 0.8],
        metadata={"help": "Confidence thresholds for logging accuracy"},
    )
    confidence_threshold: float = field(
        default=0.0,
        metadata={
            "help": "Confidence threshold for including data during training"
        },
    )
    use_flash_attention_2: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to use flash attention 2"
        },
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to use gradient checkpointing"
        },
    )
    output_dir: str = field(
        default='',
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written."
        },
    )
    dataloader_drop_last: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether to drop the last incomplete batch."
        },
    )

    initial_sensitivity: float = field(
        default=0.5,
        metadata={"help": "Initial value for sensitivity parameter (0-1)"},
    )
    initial_specificity: float = field(
        default=0.5,
        metadata={"help": "Initial value for specificity parameter (0-1)"},
    )
    ss_learning_rate: float = field(
        default=1e-2,
        metadata={"help": "Learning rate for sensitivity and specificity parameters"},
    )