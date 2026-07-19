import transformers, torch, os, datasets, random,utils
import torch.nn.functional as F, torch, torch.nn as nn
import utils.data_utils as data_utils
import geoopt
from quant.ost_model_utils import LM
from eaquant.data.medmix import get_medmix_train_texts
from eaquant.evidence.schema import (
    FaithfulnessAugmentedDataset,
    FaithfulnessDataCollator,
    load_faithfulness_training_records,
)
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainerCallback,
    default_data_collator,
)
from accelerate.hooks import remove_hook_from_module
from utils.data_utils import CustomJsonDataset, group_texts
from datasets import Dataset, IterableDataset
from quant.cayley_opt import SGDG
from torch.optim import lr_scheduler
from accelerate import DistributedType
from quant.trainer import MyTrainer
from loguru import logger
from utils import distribute_model

MEDMIX_TRAIN_CACHE_VERSION = "answer_first_rationale_v1"


def rotate_smooth_train(args, lm: LM):

    logger.info("train rotate model")
    if args.smooth_up_down:
        logger.info("train smooth up down")
    if args.smooth_up_gate:
        logger.info("train smooth up gate")
    if args.smooth_qk:
        logger.info("train smooth qk")
    if args.smooth_ov:
        logger.info("train smooth ov")
    if args.smooth_norm_linear:
        logger.info("smooth norm linear")

    lm.model.config.use_cache = False
    if args.gradient_checkpointing and hasattr(lm.model, "enable_input_require_grads"):
        lm.model.enable_input_require_grads()
    train_dataset, eval_dataset = get_train_eval_dataset(args, lm.tokenizer)
    if lm.tokenizer.pad_token is None:
        lm.tokenizer.pad_token = lm.tokenizer.eos_token
    param_keys = get_param_keys(lm.model)
    utils.cleanup_memory()
    if args.train_distribute:
        distribute_model(lm.model)
    data_collator = default_data_collator
    if args.explanation_loss_enabled:
        data_collator = FaithfulnessDataCollator(
            tokenizer=lm.tokenizer,
            max_length=args.faithfulness_max_seq_len,
            answer_scoring_mode=args.faithfulness_answer_scoring_mode,
        )
    trainer = MyTrainer(
        model=lm.model,
        tokenizer=lm.tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        args=args,
    )
    trainer.train()
    acc = trainer.accelerator
    st = {k: v for k, v in (acc.get_state_dict(trainer.model)).items() if k in param_keys}
    acc.wait_for_everyone()
    if acc.is_main_process:
        torch.save(st, f"{args.output_dir}/model.bin")
    else:
        print(f"sub process{acc.process_index} exit")
        exit(0)
    if acc.distributed_type == DistributedType.FSDP:
        print("reloading lm")
        new_lm = LM(args)
        new_lm.fuse_layer_norms()
        new_lm.generate_rotate_parameters(args)
        new_lm.model.load_state_dict(st, strict=False)
        return new_lm
    else:
        lm.model = acc.unwrap_model(trainer.model)
        if args.train_distribute:
            remove_hook_from_module(lm.model)
        return lm


def get_train_eval_dataset(args, tokenizer):
    cache_parts = ["tokenized", args.train_dataset]
    if args.train_dataset == "medmix":
        cache_parts.append(MEDMIX_TRAIN_CACHE_VERSION)
    cache_parts.append(f"seqlen{args.seqlen}")
    cache_dir = "./cache/" + args.model.split("/")[-1] + "_".join(cache_parts)
    
    
    
    
    
    
    
    
    
    if os.path.exists(cache_dir):
        tokenized_datasets = datasets.load_from_disk(cache_dir)
    else:
        if args.train_dataset == "wikitext2":
            train_dataset = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        elif args.train_dataset == "medmix":
            medmix_texts = get_medmix_train_texts()
            logger.info(f"Built medmix train source with {len(medmix_texts)} examples")
            train_dataset = Dataset.from_dict({"text": medmix_texts})
        else:
            raise ValueError(
                f"Unsupported train_dataset='{args.train_dataset}'. Currently supported: wikitext2, medmix"
            )

        def tokenize_function(examples):
            tokenized = tokenizer(examples["text"])
            return {
                key: value
                for key, value in tokenized.items()
                if key in ("input_ids", "attention_mask")
            }

        tokenized_datasets = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names,
        )
        grouped_datasets = group_texts(args.seqlen, tokenized_datasets)
        tokenized_datasets = Dataset.from_dict(grouped_datasets)
        tokenized_datasets.save_to_disk(cache_dir)
    if args.explanation_loss_enabled:
        faith_records = load_faithfulness_training_records(
            cache_path=args.faithfulness_cache_path,
            limit=args.faithfulness_cache_limit,
        )
        logger.info(
            f"Loaded {len(faith_records)} faithfulness training records from "
            f"{args.faithfulness_cache_path}"
        )
        tokenized_datasets = FaithfulnessAugmentedDataset(
            base_dataset=tokenized_datasets,
            faith_records=faith_records,
        )
    test_loader = data_utils.get_loaders(
        args.eval_dataset, seed=args.seed, model=args.model, seqlen=args.seqlen, eval_mode=True
    )
    nsample = test_loader["input_ids"].numel() // args.seqlen
    input_ids = test_loader["input_ids"].reshape(-1)[: nsample * args.seqlen]
    eval_dataset = Dataset.from_dict(dict(input_ids=input_ids.split(args.seqlen, dim=-1)))

    def f(examples):
        examples["labels"] = examples["input_ids"]
        return examples

    eval_dataset = eval_dataset.map(f)
    return tokenized_datasets, eval_dataset


def get_param_keys(model):
    keys = list()
    for k, v in model.named_parameters():
        if v.requires_grad:
            keys.append(k)
    return keys
