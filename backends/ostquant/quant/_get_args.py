import os
import torch, torch.nn as nn, torch.nn.functional as F, argparse, transformers, sys, lm_eval, pprint
from transformers import Seq2SeqTrainingArguments
from typing import Optional, List, Union
from dataclasses import dataclass, field
from loguru import logger

def parse_args():
    parser = transformers.HfArgumentParser(TrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]
    init_logger(args)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if (
        getattr(args, "ddp_find_unused_parameters", None) is None
        and (world_size > 1 or getattr(args, "local_rank", -1) != -1)
    ):
        # Training does a no-grad teacher forward before the trainable forward,
        # so DDP needs explicit unused-parameter detection.
        args.ddp_find_unused_parameters = True
    if getattr(args, "gradient_checkpointing", False):
        args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    if (args.online_hadamard == "all" or args.online_hadamard == "v") and args.rotate_ov:
        logger.warning("on_line_hadamard and rotate_ov are both enabled, we will disalbe rotate_ov")
        args.rotate_ov = False
    if args.rotate_post_rope and args.online_qk_hadamard:
        logger.warning("online_qk_hadamard and rotate_post_rope are both enabled, we will disalbe online_qk_hdamard")
        args.online_qk_hadamard = False
    if args.explanation_loss_enabled and not args.faithfulness_cache_path:
        raise ValueError(
            "explanation_loss_enabled=True requires --faithfulness_cache_path."
        )
    if args.explanation_loss_enabled and args.train_dataset != "medmix":
        raise ValueError(
            "Explanation-aware loss is currently supported only when train_dataset=medmix."
        )
    if args.explanation_loss_enabled and args.remove_unused_columns:
        logger.warning(
            "Explanation-aware training requires remove_unused_columns=False so "
            "_faith_record reaches FaithfulnessDataCollator. Overriding the "
            "default Trainer column-pruning behavior."
        )
        args.remove_unused_columns = False

    args.embed_quant_params = dict(bits=args.residual_bits, sym=not args.a_asym, dynamic=True, dynamic_method="perchannel")
    
    args.weight_quant_params = dict(
        bits=args.w_bits,
        sym=not args.w_asym,
        groupsize=args.w_groupsize,
        dynamic=True,
        dynamic_method="pertoken",
        mse=args.w_clip,
    )
    
    args.norm_quant_params = dict(
        bits=args.a_bits,
        sym=not args.a_asym,
        groupsize=args.a_groupsize,
        clip_ratio=args.a_clip_ratio,
        dynamic=True,
        dynamic_method=args.a_dynamic_method,
    )  
    args.ropek_quant_params = dict(
        bits=args.k_bits,
        sym=not args.k_asym,
        groupsize=args.k_groupsize,
        clip_ratio=args.k_clip_ratio,
        dynamic=True,
        dynamic_method=args.a_dynamic_method,
    )  
    args.v_proj_quant_params = dict(
        bits=args.v_bits,
        sym=not args.v_asym,
        groupsize=args.v_groupsize,
        clip_ratio=args.v_clip_ratio,
        dynamic=True,
        dynamic_method=args.a_dynamic_method,
    )  
    args.pv_matmul_quant_params = dict(
        bits=args.a_bits,
        sym=not args.a_asym,
        groupsize=128,
        clip_ratio=args.k_clip_ratio,
        dynamic=True,
        dynamic_method=args.a_dynamic_method,
    )  
    args.mul_quant_params = dict(
        bits=args.down_bits,
        sym=not args.a_asym,
        groupsize=args.a_groupsize,
        clip_ratio=args.k_clip_ratio,
        dynamic=True,
        dynamic_method=args.a_dynamic_method,
    )  
    
    
    args.q_proj_quant_params = dict(
        bits=args.a_bits,
        sym=not args.a_asym,
        groupsize=args.a_groupsize,
        clip_ratio=args.a_clip_ratio,
        dynamic=True,
        dynamic_method=args.a_dynamic_method,
    )  
    args.ropeq_quant_params = dict(
        bits=args.a_bits,
        sym=not args.a_asym,
        groupsize=args.a_groupsize,
        clip_ratio=args.a_clip_ratio,
        dynamic=True,
        dynamic_method=args.a_dynamic_method,
    )  
    args.k_proj_quant_params = dict(
        bits=args.a_bits,
        sym=not args.k_asym,
        groupsize=args.a_groupsize,
        clip_ratio=args.a_clip_ratio,
        dynamic=True,
        dynamic_method=args.a_dynamic_method,
    )  
    args.qk_matmul_quant_params = dict(
        bits=args.attn_bits,
        sym=not args.a_asym,
        groupsize=args.a_groupsize,
        clip_ratio=args.a_clip_ratio,
        dynamic=True,
        dynamic_method=args.a_dynamic_method,
    )  
    args.softmax_quant_params = dict(
        bits=args.attn_bits,
        sym=not args.a_asym,
        groupsize=args.a_groupsize,
        clip_ratio=args.a_clip_ratio,
        dynamic=True,
        dynamic_method=args.a_dynamic_method,
    )  
    args.o_proj_quant_params = dict(
        bits=args.residual_bits,
        sym=not args.a_asym,
        groupsize=args.a_groupsize,
        clip_ratio=args.a_clip_ratio,
        dynamic=True,
        dynamic_method="perchannel",
    )
    args.resadd1_quant_params = dict(
        bits=args.residual_bits,
        sym=not args.a_asym,
        groupsize=args.a_groupsize,
        clip_ratio=args.a_clip_ratio,
        dynamic=True,
        dynamic_method="perchannel",
    )
    args.up_proj_quant_params = dict(
        bits=args.a_bits,
        sym=not args.a_asym,
        groupsize=args.a_groupsize,
        clip_ratio=args.k_clip_ratio,
        dynamic=True,
        dynamic_method=args.a_dynamic_method,
    )  
    args.gate_proj_quant_params = dict(
        bits=args.act_bits,
        sym=not args.a_asym,
        groupsize=args.a_groupsize,
        clip_ratio=args.k_clip_ratio,
        dynamic=True,
        dynamic_method=args.a_dynamic_method,
    )  
    args.silu_quant_params = dict(
        bits=args.act_bits,
        sym=not args.a_asym,
        groupsize=args.a_groupsize,
        clip_ratio=args.k_clip_ratio,
        dynamic=True,
        dynamic_method=args.a_dynamic_method,
    )  
    args.down_proj_quant_params = dict(
        bits=args.residual_bits,
        sym=not args.a_asym,
        groupsize=args.a_groupsize,
        clip_ratio=args.k_clip_ratio,
        dynamic=True,
        dynamic_method="perchannel",
    )  
    args.resadd2_quant_params = dict(
        bits=args.residual_bits,
        sym=not args.a_asym,
        groupsize=args.a_groupsize,
        clip_ratio=args.k_clip_ratio,
        dynamic=True,
        dynamic_method="perchannel",
    )
    logger.debug(f"Arguments:{pprint.pformat(vars(args))}")
    logger.debug("--" * 30)
    return args


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    
    
    model: str = field(default="weights/llama3-8b-hf", metadata=dict(help="Path to the model"))
    seqlen: int = field(default=2048, metadata=dict(help="Sequence length for calibration, training, and evaluation"))
    fully_quant: bool = field(default=False, metadata=dict(help="Whether to use fully quantization"))
    test_static: bool = field(default=False, metadata=dict(help="Whether to use static quantization for activation"))
    use_sdpa: bool = field(default=True, metadata=dict(help="Use SDPA instead of default MHA"))

    train_distribute:bool = field(default=False, metadata=dict(help="Whether to use distributed training"))
    train_rotate: bool = field(default=True, metadata=dict(help="Whether to train with rotation"))
    max_steps: int = field(default=200, metadata=dict(help="Maximum number of training steps"))
    loss_type: str = field(
        default="origin",
        metadata=dict(
            choices=[
                "origin",
                "mse",
                "kl",
                "kd",
                "feature_mse",
                "r_kl_top",
                "rkl",
                "kl_top",
                "kl_top_5",
                "kl_top_10",
                "kl_top_50",
                "kl_top_100",
                "kl_top_500",
            ],
            help="Loss type for training",
        ),
    )
    opt_type: str = field(default="RAdam", metadata=dict(choices=["SGDG", "RAdam", "RSGD"], help="Optimizer type for training"))
    rotate_lr: float = field(default=0.01689753172873217, metadata=dict(help="Learning rate for rotation"))
    smooth_lr: float = field(default=0.0017898822675917248, metadata=dict(help="Learning rate for smoothing"))
    rotate_momentom: float = field(default=0, metadata=dict(help="Momentum for rotation"))
    smooth_momentom: float = field(default=0.9, metadata=dict(help="Momentum for smoothing"))
    train_enable_wquant: bool = field(default=False, metadata=dict(help="Enable weight quantization during training"))
    train_dataset: str = field(
        default="wikitext2",
        metadata=dict(choices=["wikitext2", "c4", "pdb", "medmix"], help="Dataset for training"),
    )
    explanation_loss_enabled: bool = field(
        default=False,
        metadata=dict(help="Enable explanation-aware faithfulness losses during train_rotate."),
    )
    faithfulness_cache_path: Optional[str] = field(
        default=None,
        metadata=dict(help="Path to the medmix teacher-faithfulness jsonl cache."),
    )
    faithfulness_cache_limit: Optional[int] = field(
        default=None,
        metadata=dict(help="Optional cap on the number of faithfulness-cache records used for training."),
    )
    faithfulness_max_seq_len: int = field(
        default=1024,
        metadata=dict(help="Maximum sequence length for explanation/recovery loss branches."),
    )
    faithfulness_answer_scoring_mode: str = field(
        default="letter_option_mean_logprob",
        metadata=dict(
            choices=["single_letter", "letter_option_mean_logprob"],
            help=(
                "How recovery-answer options are represented in the faithfulness "
                "cache builder and training objective."
            ),
        ),
    )
    explanation_token_loss_weight: float = field(
        default=0.1,
        metadata=dict(help="Weight for the selected-span token KL on the full teacher rationale."),
    )
    explanation_recovery_loss_weight: float = field(
        default=0.1,
        metadata=dict(help="Weight for the selected-rationale option-set KL loss."),
    )
    resume_path: str = field(default=None, metadata=dict(help="Path to resume training"))
    
    per_device_train_batch_size: int = field(default=8, metadata=dict(help="Batch size per device for training"))
    per_device_eval_batch_size: int = field(default=8, metadata=dict(help="Batch size per device for evaluation"))
    
    
    rotate: bool = field(default=True, metadata=dict(help="Whether to rotate the model"))
    rotate_mode: str = field(default="hadamard", metadata=dict(choices=["hadamard", "random"], help="Initial rotation mode"))
    rotate_ov: bool = field(default=False, metadata=dict(help="Rotate V's output and O's input"))
    rotate_pre_rope: bool = field(default=False, metadata=dict(help="Rotate ROPE's input"))
    rotate_post_rope: bool = field(default=False, metadata=dict(help="Rotate ROPE's output"))
    rotate_rope_perlayer: bool = field(default=True, metadata=dict(help="Whether to allow each layer to have a separate ROPE matrix"))
    smooth_up_down: bool = field(default=False, metadata=dict(help="Smooth Up's output and Down's output"))
    smooth_up_gate: bool = field(default=False, metadata=dict(help="Smooth x1 and x2"))
    smooth_qk: bool = field(default=False, metadata=dict(help="Smooth q and k"))
    smooth_ov: bool = field(default=False, metadata=dict(help="Smooth v and o"))
    smooth_norm_linear: bool = field(default=False, metadata=dict(help="Smooth norm and linear after rotation"))

    rotate_down_dim: int = field(default=1, metadata=dict(help="Dimension for rotating down projection"))
    fp32_had: bool = field(default=True, metadata=dict(help="Apply Hadamard rotation in FP32"))
    online_hadamard: str = field(
        default="down",
        metadata=dict(choices=["all", "v", "down", "None"], help="Online Hadamard transformation settings"),
    )
    online_qk_hadamard: bool = field(default=True, metadata=dict(help="Apply online Hadamard to Q/K"))
    
    a_bits: int = field(default=4, metadata=dict(help="Number of bits for inputs of linear layers"))
    a_dynamic_method: str = field(
        default="pertoken",
        metadata=dict(choices=["pertoken", "perchannel", "pertensor"], help="Dynamic quantization method"),
    )
    a_groupsize: int = field(default=-1, metadata=dict(help="Groupsize for activation quantization"))
    a_asym: bool = field(default=True, metadata=dict(help="Asymmetric activation quantization"))
    a_clip_ratio: float = field(default=1.0, metadata=dict(help="Clip ratio for activation quantization"))
    a_static: bool = field(default=False, metadata=dict(help="Whether to use static quantization for activation"))
    
    w_bits: int = field(default=4, metadata=dict(help="Number of bits for weights of linear layers"))
    w_groupsize: int = field(default=-1, metadata=dict(help="Groupsize for weight quantization"))
    w_asym: bool = field(default=False, metadata=dict(help="Asymmetric weight quantization"))
    w_clip: bool = field(default=True, metadata=dict(help="Clip weights during quantization"))
    force_clip: bool = field(default=True, metadata=dict(help="Clip weights during GPTQ"))
    w_gptq: bool = field(default=True, metadata=dict(help="Use GPTQ for weight quantization"))
    nsamples: int = field(default=128, metadata=dict(help="Number of calibration data samples for GPTQ"))
    cal_dataset: str = field(
        default="wikitext2",
        metadata=dict(choices=["wikitext2", "pdb", "c4", "medmix"], help="Calibration dataset for GPTQ"),
    )
    percdamp: float = field(default=0.01, metadata=dict(help="Percent of the average Hessian diagonal to use for dampening"))
    act_order: bool = field(default=False, metadata=dict(help="Activation order in GPTQ"))

    
    v_bits: int = field(default=4, metadata=dict(help="Number of bits for V-cache quantization"))
    v_groupsize: int = field(default=128, metadata=dict(help="Groupsize for V-cache quantization"))
    v_asym: bool = field(default=True, metadata=dict(help="Asymmetric V-cache quantization"))
    v_clip_ratio: float = field(default=1.0, metadata=dict(help="Clip ratio for V-cache quantization"))
    k_bits: int = field(default=4, metadata=dict(help="Number of bits for K-cache quantization"))
    k_groupsize: int = field(default=128, metadata=dict(help="Groupsize for K-cache quantization"))
    k_asym: bool = field(default=True, metadata=dict(help="Asymmetric K-cache quantization"))
    k_clip_ratio: float = field(default=1.0, metadata=dict(help="Clip ratio for K-cache quantization"))
    
    residual_bits: int = field(default=16, metadata=dict(help="Bits for residual inputs and outputs"))
    attn_bits: int = field(default=16, metadata=dict(help="Bits for attention outputs"))
    act_bits: int = field(default=16, metadata=dict(help="Bits for activation"))
    down_bits: int = field(default=4, metadata=dict(help="Bits for down projection input"))
    int8_down_proj:bool = field(default=False, metadata=dict(help="Whether to use int8 for down projection"))

    
    load_qmodel_path: Optional[str] = field(default=None, metadata=dict(help="Path to load the quantized model"))
    save_qmodel_path: Optional[str] = field(default=None, metadata=dict(help="Path to save the quantized model"))

    
    eval_dataset: str = field(default="wikitext2", metadata=dict(choices=["wikitext2", "pdb", "c4"], help="Dataset for evaluation"))
    bsz: int = field(default=4, metadata=dict(help="Batch size for evaluation"))
    skip_final_eval: bool = field(default=False, metadata=dict(help="Skip final perplexity evaluation after quantization"))
    lm_eval: bool = field(default=False, metadata=dict(help="Evaluate the model on LM Eval tasks"))
    tasks: List[str] = field(
        default_factory=lambda: [
            "arc_challenge",
            "arc_easy",
            "boolq",
            "hellaswag",
            "lambada_openai",
            "openbookqa",
            "piqa",
            "social_iqa",
            "winogrande",
        ],
        metadata=dict(help="List of LM Eval tasks"),
    )
    lm_eval_batch_size: int = field(default=16, metadata=dict(help="Batch size for evaluating with LM Eval harness"))
    distribute: bool = field(default=False, metadata=dict(help="Distribute the model on multiple GPUs for evaluation"))

    pre_eval: bool = field(default=False, metadata=dict(help="Whether to evaluate the model before training"))
    
    qwen2_downfill: bool = field(default=False, metadata=dict(help="Whether to fill the down projection with zeros"))
    gradient_checkpointing: bool = field(default=True, metadata=dict(help="Whether to use gradient checkpointing"))
    logging_steps: int = field(default=1, metadata=dict(help="Number of steps between logging"))
    log_on_each_node: bool = field(default=False, metadata=dict(help="Whether to log on each node"))
    fp16: bool = field(default=False, metadata=dict(help="Whether to use fp16"))
    eval_strategy: str = field(default="steps", metadata=dict(choices=["no", "steps", "epoch"], help="Evaluation strategy"))
    eval_steps: float = field(default=0.5, metadata=dict(help="Number of steps between evaluation"))
    save_strategy: str = field(default="no", metadata=dict(choices=["steps", "epoch", "no"], help="Save strategy"))
    fsdp_config: Optional[Union[dict, str]] = field(
        default_factory=lambda: {
            "cpu_ram_efficient_loading": True,
            
            "transformer_layer_cls_to_wrap": ["QuantDecoderLayer","QuantLinear"]
        },
        metadata={
            "help": (
                "Config to be used with FSDP (Pytorch Fully Sharded  Data Parallel). The value is either a "
                "fsdp json config file (e.g., `fsdp_config.json`) or an already loaded json file as `dict`."
            )
        },
    )
    report_to:str = field(default="none", metadata=dict(choices=["none", "wandb", "tensorboard"], help="Where to report metrics"))
    force_rdtype_inplace: bool = field(default=False,metadata=dict(help="when inplace weather use rtype if not we use fp64 to merge weights"))
    use_klt: bool = field(default=False,metadata=dict(help="whether to use klt"))


    sub_mean:bool = field(default=True,metadata=dict(help="whether to use sub mean"))
    post_attn: bool = field(default=False,metadata=dict(help="whether to use post attn for calculate kl loss"))

def init_logger(args):
    logger.remove()
    if args.local_rank in (0, -1):
        logger.add(sys.stdout, level="INFO")
        output_dir = args.output_dir
        log_file = output_dir + "/log.txt"
        logger.add(log_file, level="DEBUG")


if __name__ == "__main__":
    args = parse_args()
    print(args)
