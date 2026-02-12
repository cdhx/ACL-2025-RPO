# Copyright 2024 the LlamaFactory team.
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

import os
from functools import partial
from typing import TYPE_CHECKING, Any, Dict, List, Union

from datasets import Features

from ..extras.logging import get_logger
from .data_utils import Role


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import Seq2SeqTrainingArguments

    from ..hparams import DataArguments
    from .parser import DatasetAttr


logger = get_logger(__name__)


def _convert_images(images: List[Any], dataset_attr: "DatasetAttr", data_args: "DataArguments") -> List[Any]:
    r"""
    Optionally concatenates image path to dataset dir when loading from local disk.
    """
    outputs = []
    if dataset_attr.load_from in ["script", "file"]:
        for image in images:
            if isinstance(image, str) and os.path.isfile(os.path.join(data_args.dataset_dir, image)):
                outputs.append(os.path.join(data_args.dataset_dir, image))
            else:
                outputs.append(image)

    return outputs


def convert_alpaca(
    examples: Dict[str, List[Any]], dataset_attr: "DatasetAttr", data_args: "DataArguments"
) -> Dict[str, List[Any]]:
    r"""
    Converts alpaca format dataset to the standard format.
    """
    outputs = {"prompt": [], "response": [], "system": [], "tools": [], "images": []}
    convert_images = partial(_convert_images, dataset_attr=dataset_attr, data_args=data_args)
    for i in range(len(examples[dataset_attr.prompt])):
        prompt = []
        if dataset_attr.history and isinstance(examples[dataset_attr.history][i], list):
            for old_prompt, old_response in examples[dataset_attr.history][i]:
                prompt.append({"role": Role.USER.value, "content": old_prompt})
                prompt.append({"role": Role.ASSISTANT.value, "content": old_response})

        content = []
        if dataset_attr.prompt and examples[dataset_attr.prompt][i]:
            content.append(examples[dataset_attr.prompt][i])

        if dataset_attr.query and examples[dataset_attr.query][i]:
            content.append(examples[dataset_attr.query][i])

        prompt.append({"role": Role.USER.value, "content": "\n".join(content)})  # "prompt\nquery"

        if dataset_attr.kto_tag and isinstance(examples[dataset_attr.kto_tag][i], bool):  # kto example
            response = [{"role": Role.ASSISTANT.value, "content": examples[dataset_attr.response][i]}]
            if examples[dataset_attr.kto_tag][i]:
                response = response + [{"role": Role.ASSISTANT.value, "content": ""}]
            else:
                response = [{"role": Role.ASSISTANT.value, "content": ""}] + response
        elif (
            dataset_attr.ranking
            and isinstance(examples[dataset_attr.chosen][i], str)
            and isinstance(examples[dataset_attr.rejected][i], str)
        ):  # pairwise example
            response = [
                {"role": Role.ASSISTANT.value, "content": examples[dataset_attr.chosen][i]},
                {"role": Role.ASSISTANT.value, "content": examples[dataset_attr.rejected][i]},
            ]
        elif dataset_attr.response and isinstance(examples[dataset_attr.response][i], str):  # normal example
            response = [{"role": Role.ASSISTANT.value, "content": examples[dataset_attr.response][i]}]
        else:  # unsupervised
            response = []

        outputs["prompt"].append(prompt)
        outputs["response"].append(response)
        outputs["system"].append(examples[dataset_attr.system][i] if dataset_attr.system else "")
        outputs["tools"].append(examples[dataset_attr.tools][i] if dataset_attr.tools else "")
        outputs["images"].append(convert_images(examples[dataset_attr.images][i]) if dataset_attr.images else [])

    return outputs


def convert_sharegpt(
    examples: Dict[str, List[Any]], dataset_attr: "DatasetAttr", data_args: "DataArguments"
) -> Dict[str, List[Any]]:
    r"""
    Converts sharegpt format dataset to the standard format.
    """
    # examples是lazybatch对象，就是本地文件加载进来，按照key组成的conversation:[],chosen:[],reject:[],sid:[],tid:[]这样的结构，每个里面是1000个元素    
    # TODO_DONE margin
    outputs = {"prompt": [], "response": [], "system": [], "tools": [], "images": []}
    if 'margin' in examples.keys():
        outputs["margin"] = []
    
        
    convert_images = partial(_convert_images, dataset_attr=dataset_attr, data_args=data_args)
    tag_mapping = {
        dataset_attr.user_tag: Role.USER.value,
        dataset_attr.assistant_tag: Role.ASSISTANT.value,
        dataset_attr.observation_tag: Role.OBSERVATION.value,
        dataset_attr.function_tag: Role.FUNCTION.value,
        dataset_attr.system_tag: Role.SYSTEM.value,
    }
    odd_tags = (dataset_attr.user_tag, dataset_attr.observation_tag)
    even_tags = (dataset_attr.assistant_tag, dataset_attr.function_tag)
    accept_tags = (odd_tags, even_tags)
    # 逐个conversations对象处理，examples[dataset_attr.messages]是一个长度为1000的conversations列表
    for i, messages in enumerate(examples[dataset_attr.messages]):
        if len(messages) == 0:
            continue
        # 提取system，如果第一个对话历史是system的tag，就把第一个当成system，然后把system从messages里删掉
        if dataset_attr.system_tag and messages[0][dataset_attr.role_tag] == dataset_attr.system_tag:
            system = messages[0][dataset_attr.content_tag]
            messages = messages[1:]
        # 否则一个个是有一个专门的system的字段，从这里获取system
        else:
            system = examples[dataset_attr.system][i] if dataset_attr.system else ""

        aligned_messages = []
        broken_data = False
        # 逐个conversations对象处理，似乎只是把conversations放到align_messages里，从from value，变成role content，然后记录一下是不是broken_data
        for turn_idx, message in enumerate(messages):
            # 检查conversations里，每轮的role是不是合法的，奇数是哪些，偶数是哪些
            if message[dataset_attr.role_tag] not in accept_tags[turn_idx % 2]:
                logger.warning("Invalid role tag in {}.".format(messages))
                broken_data = True
            # 都放在aligned_messages这个数据里
            aligned_messages.append(
                {"role": tag_mapping[message[dataset_attr.role_tag]], "content": message[dataset_attr.content_tag]}
            )
        # align messages（对话历史）的数量做check，要么不是ranking数据，那就是sft数据，对话历史应该是偶数个，如果是ranking，那就应该是奇数（response单独拿出来了）
        if (not dataset_attr.ranking and len(aligned_messages) % 2 != 0) or (
            dataset_attr.ranking and len(aligned_messages) % 2 == 0
        ):
            logger.warning("Invalid message count in {}.".format(messages))
            broken_data = True
        # 处理kto，现在用不到
        if dataset_attr.kto_tag and isinstance(examples[dataset_attr.kto_tag][i], bool):  # kto example
            prompt = aligned_messages[:-1]
            response = aligned_messages[-1:]
            if examples[dataset_attr.kto_tag][i]:
                response = response + [{"role": Role.ASSISTANT.value, "content": ""}]
            else:
                response = [{"role": Role.ASSISTANT.value, "content": ""}] + response
        # ranking数据
        elif (
            dataset_attr.ranking
            and isinstance(examples[dataset_attr.chosen][i], dict)
            and isinstance(examples[dataset_attr.rejected][i], dict)
        ):  # pairwise example
            # 获取chosen和reject
            chosen = examples[dataset_attr.chosen][i]
            rejected = examples[dataset_attr.rejected][i]
            if (
                chosen[dataset_attr.role_tag] not in accept_tags[-1]
                or rejected[dataset_attr.role_tag] not in accept_tags[-1]
            ):
                logger.warning("Invalid role tag in {}.".format([chosen, rejected]))
                broken_data = True
            # rankning数据的prompt就是历史，没有system的conversations
            # response就是chosen和reject，放在一个list里
            prompt = aligned_messages
            response = [
                {"role": tag_mapping[chosen[dataset_attr.role_tag]], "content": chosen[dataset_attr.content_tag]},
                {"role": tag_mapping[rejected[dataset_attr.role_tag]], "content": rejected[dataset_attr.content_tag]},
            ]
        # 除了kto，ranking数据，那这个就应该是处理sft数据的了
        else:  # normal example
            # sft数据就是prompt是除了最后一轮回复之外的历史，reponse就是最后一个历史
            prompt = aligned_messages[:-1]
            response = aligned_messages[-1:]
        # 如果数据有问题，不记录
        if broken_data:
            logger.warning("Skipping this abnormal example.")
            continue

        outputs["prompt"].append(prompt)
        outputs["response"].append(response)
        outputs["system"].append(system)
        outputs["tools"].append(examples[dataset_attr.tools][i] if dataset_attr.tools else "")
        outputs["images"].append(convert_images(examples[dataset_attr.images][i]) if dataset_attr.images else [])
        # margin TODO_DONE 
        if 'margin' in examples.keys():
            outputs["margin"].append(examples['margin'][i])

    return outputs


def align_dataset(
    dataset: Union["Dataset", "IterableDataset"],
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
) -> Union["Dataset", "IterableDataset"]:
    r"""
    Aligned dataset:
        prompt: [{"role": "user", "content": "..."}] * (2T - 1)
        response: [{"role": "assistant", "content": "..."}] * N (N > 1 for ranking dataset)
        system: "..."
        tools: "...",
        images: [],
    """
    if dataset_attr.formatting == "alpaca":
        convert_func = partial(convert_alpaca, dataset_attr=dataset_attr, data_args=data_args)
    else:
        convert_func = partial(convert_sharegpt, dataset_attr=dataset_attr, data_args=data_args)

    column_names = list(next(iter(dataset)).keys())
    # TODO_DONE margin,这个不能简单地字典赋值
    if 'margin' in column_names:
        features = Features.from_dict(
            {
            "prompt": [
                {"role": {"dtype": "string", "_type": "Value"}, "content": {"dtype": "string", "_type": "Value"}}
            ],
            "response": [
                {"role": {"dtype": "string", "_type": "Value"}, "content": {"dtype": "string", "_type": "Value"}}
            ],
            "system": {"dtype": "string", "_type": "Value"},
            "tools": {"dtype": "string", "_type": "Value"},
            "images": [{"_type": "Image"}],        
            "margin": {"dtype": "int64", "_type": "Value"}, # TODO_DONE margin
            }
        )
    else:
        features = Features.from_dict(
            {
                "prompt": [
                    {"role": {"dtype": "string", "_type": "Value"}, "content": {"dtype": "string", "_type": "Value"}}
                ],
                "response": [
                    {"role": {"dtype": "string", "_type": "Value"}, "content": {"dtype": "string", "_type": "Value"}}
                ],
                "system": {"dtype": "string", "_type": "Value"},
                "tools": {"dtype": "string", "_type": "Value"},
                "images": [{"_type": "Image"}],         
            }
        )
    kwargs = {}
    # 应用convert_func对应的函数转换数据，比如conver_sharegpt
    if not data_args.streaming:
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
            desc="Converting format of dataset",
        )

    return dataset.map(
        convert_func,
        batched=True,
        remove_columns=column_names, #　所有列都去掉
        features=features, # 返回结果的数据结构
        **kwargs,
    )
