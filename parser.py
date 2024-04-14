import argparse
import deepspeed
from transformers import SchedulerType

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ml-1m', type=str)
    parser.add_argument('--finetune_dataset', default='ml-1m', type=str)
    parser.add_argument('--eval_k', default=[1,5,10], help='', type=list)
    parser.add_argument('--epoch', default=1, help='epoch', type=int)
    parser.add_argument('--learning_rate', default=0.0001, help='learning rate', type=float)
    parser.add_argument('--batch_size', default=512, help='batch size', type=int)
    parser.add_argument('--fewshot_size', default=512, help='batch size', type=int)
    parser.add_argument('--mask_prob', default=0.15, help='masking probability', type=float)
    parser.add_argument('--seq_length', default=50, help='length of input sequence', type=int)
    parser.add_argument('--num_header', default=2, help='number of header', type=int)
    parser.add_argument('--hidden_units', default=32, help='header dim', type=int)
    parser.add_argument('--intermediate_dim', default=256, help='intermediate dim', type=int)
    parser.add_argument('--num_layer', default=2, help='number of layer', type=int)
    parser.add_argument('--mode', default='train', choices=['train', 'svdtrain', 'freezetrain'])
    parser.add_argument('--action', default='none', choices=['none', 'fewshot'])
    parser.add_argument('--init_model_path', default='none', type=str)
    parser.add_argument('--model', default='recppt', choices=['bert4rec', 'sas4rec', 'recppt'])
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--resume_epoch', default=0, type=int)
    parser.add_argument('--val_step', type=int, default=5000)
    parser.add_argument('--save_step', type=int, default=5000)
    # deepspeed features
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument(
                        '--zero_stage',
                        type=int,
                        default=1,
                        help='ZeRO optimization stage for Actor model (and clones).')
    parser.add_argument(
                        "--per_device_eval_batch_size",
                        type=int,
                        default=256,
                        help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument('--data_path',
                        nargs='*',
                        default=['Dahoas/rm-static'],
                        help='Path to the training dataset. Accepted format:'
                        '1) a single data path, 2) multiple datasets in the'
                        'form: dataset1-path dataset2-path ...')
    parser.add_argument('--data_split',
                        type=str,
                        default='6,2,2',
                        help='Comma-separated list of proportions for training'
                        'phase 1, 2, and 3 data. For example the split `2,4,4`'
                        'will use 60% of data for phase 1, 20% for phase 2'
                        'and 20% for phase 3.')
    parser.add_argument(
                        '--data_output_path',
                        type=str,
                        default='/tmp/data_files/',
                        help='Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)')
    parser.add_argument(
                        "--model_name_or_path",
                        type=str,
                        help="Path to pretrained model or model identifier from huggingface.co/models.",
                        required=False)
    parser.add_argument(
                        "--per_device_train_batch_size",
                        type=int,
                        default=256,
                        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.1,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
                        "--gradient_accumulation_steps",
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument(
                        "--lr_scheduler_type",
                        type=SchedulerType,
                        default="cosine",
                        help="The scheduler type to use.",
                        choices=[
                            "linear", "cosine", "cosine_with_restarts", "polynomial",
                            "constant", "constant_with_warmup"])
    parser.add_argument(
                        "--num_warmup_steps",
                        type=int,
                        default=20,
                        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=4321,
                        help="A seed for reproducible training.")
    parser.add_argument('--gradient_checkpointing',
                        action='store_true',
                        help='Enable HF gradient checkpointing for model.')
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument("--lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()
