# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "trl @ git+https://github.com/huggingface/trl.git",
#     "peft",
#     "trackio",
# ]
# ///

"""
对 `open-r1/OpenR1-Math-220k` 做 on-policy 蒸馏（GKD）。

和 `trl/experimental/gold/gold.py` 保持同样的“脚本入口”风格：
- TrlParser((ScriptArguments, GKDConfig, ModelConfig, ...))
- load_dataset()
- Trainer(...).train()

特别说明：
- 数据集只用 `problem` 列作为 prompt（on-policy distillation 只需要 prompt）
- 默认建议配合 `--lmbda 1.0`，让每一步都用 student 生成（纯 on-policy）
- 不需要 test/eval：直接 `--eval_strategy no`
- “每 50 个 epoch 存一次”：用 `--save_every_n_epochs 50`
"""

from __future__ import annotations

from dataclasses import dataclass, field

from datasets import load_dataset
from transformers import AutoTokenizer, GenerationConfig, TrainerCallback, TrainerControl, TrainerState

from trl import (
    LogCompletionsCallback,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.experimental.gkd import GKDConfig, GKDTrainer


# 为了兼容一些环境里 tokenizer 没有 chat_template 的情况，提供一个最小模板
SIMPLE_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{message['role'].capitalize() + ': ' + message['content'] + '\\n\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"
)


@dataclass
class OpenR1Math220kArguments:
    problem_column: str = field(
        default="problem",
        metadata={"help": "数据集中作为 prompt 的列名（OpenR1-Math-220k 默认是 `problem`）。"},
    )
    prompt_template: str = field(
        default="Solve the following math problem. Give your final answer at the end.\n\n{problem}",
        metadata={"help": "把 problem 渲染为 prompt 的模板，必须包含 `{problem}` 占位符。"},
    )
    system_prompt: str | None = field(
        default=None,
        metadata={"help": "可选的 system prompt（会作为 messages 的第一条 system 消息）。"},
    )
    save_every_n_epochs: int = field(
        default=50,
        metadata={"help": "每 N 个 epoch 保存一次 checkpoint（<=0 表示关闭）。默认 50。"},
    )


class SaveEveryNEpochsCallback(TrainerCallback):
    def __init__(self, every_n_epochs: int):
        self.every_n_epochs = int(every_n_epochs)

    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if self.every_n_epochs <= 0 or state.epoch is None:
            return control
        epoch_int = int(state.epoch)
        if epoch_int > 0 and (epoch_int % self.every_n_epochs) == 0:
            control.should_save = True
        return control


def _to_messages(
    example: dict,
    *,
    problem_column: str,
    prompt_template: str,
    system_prompt: str | None,
) -> dict:
    problem = example.get(problem_column, None)
    if problem is None:
        raise KeyError(f"样本缺少列 `{problem_column}`，可用列：{list(example.keys())[:50]}")
    prompt = prompt_template.format(problem=problem)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    # 末尾留一个空 assistant turn，满足 ChatML collator 的格式期望；
    # 真正训练时建议 `--lmbda 1.0`，让 trainer 用 student on-policy 生成覆盖 labels。
    messages.append({"role": "assistant", "content": ""})
    return {"messages": messages}


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, GKDConfig, ModelConfig, OpenR1Math220kArguments))
    script_args, training_args, model_args, data_args = parser.parse_args_and_config()

    ################
    # Model & Tokenizer（结构对齐 gold.py）
    ################
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=model_args.dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs

    teacher_model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=model_args.dtype,
        use_cache=True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.teacher_model_init_kwargs = teacher_model_kwargs

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "chat_template", None) is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    ################
    # Dataset（只用 problem 列）
    ################
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    train_split = script_args.dataset_train_split

    def mapper(ex):
        return _to_messages(
            ex,
            problem_column=data_args.problem_column,
            prompt_template=data_args.prompt_template,
            system_prompt=data_args.system_prompt,
        )

    train_ds = dataset[train_split].map(
        mapper,
        remove_columns=dataset[train_split].column_names,
        desc=f"Map `{script_args.dataset_name}`[{train_split}] -> ChatML(messages) from `{data_args.problem_column}`",
    )

    eval_ds = None
    if training_args.eval_strategy != "no":
        test_split = script_args.dataset_test_split
        eval_ds = dataset[test_split].map(
            mapper,
            remove_columns=dataset[test_split].column_names,
            desc=f"Map `{script_args.dataset_name}`[{test_split}] -> ChatML(messages) from `{data_args.problem_column}`",
        )

    ################
    # Training
    ################
    trainer = GKDTrainer(
        model=model_args.model_name_or_path,
        teacher_model=training_args.teacher_model_name_or_path,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    if data_args.save_every_n_epochs and data_args.save_every_n_epochs > 0:
        trainer.add_callback(SaveEveryNEpochsCallback(data_args.save_every_n_epochs))

    if training_args.eval_strategy != "no":
        generation_config = GenerationConfig(
            max_new_tokens=training_args.max_new_tokens, do_sample=True, temperature=training_args.temperature
        )
        completions_callback = LogCompletionsCallback(trainer, generation_config, num_prompts=8)
        trainer.add_callback(completions_callback)

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


