# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Create CoSENT model for text matching task
"""

import math
import os

import pandas as pd
import torch
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModel, BertModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from text2vec.cosent_dataset import CosentTrainDataset, load_cosent_train_data, HFCosentTrainDataset
from text2vec.utils.stats_util import compute_spearmanr, compute_pearsonr

from text2vec.text_matching_dataset import TextMatchingTestDataset, load_text_matching_test_data, \
    HFTextMatchingTestDataset
from text2vec.utils.stats_util import set_seed
from torch.nn.parallel import DistributedDataParallel
from enum import Enum

from examples.data_sample import DistributedSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderType(Enum):
    FIRST_LAST_AVG = 0
    LAST_AVG = 1
    CLS = 2
    POOLER = 3
    MEAN = 4

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return EncoderType[s]
        except KeyError:
            raise ValueError()


class EncodeModel(nn.Module):
    def __init__(
            self,
            model_name_or_path: str = "shibing624/text2vec-base-chinese",
            encoder_type="MEAN",
            max_seq_length: int = 256,
    ):
        """
        Initializes the base sentence model.

        :param model_name_or_path: The name of the model to load from the huggingface models library.
        :param encoder_type: The type of encoder to use, See the EncoderType enum for options:
            FIRST_LAST_AVG, LAST_AVG, CLS, POOLER(cls + dense), MEAN(mean of last_hidden_state)
        :param max_seq_length: The maximum sequence length.
        :param device: Device (like 'cuda' / 'cpu') that should be used for computation. If None, checks if GPU.

        bert model: https://huggingface.co/transformers/model_doc/bert.html?highlight=bert#transformers.BertModel.forward
        BERT return: <last_hidden_state>, <pooler_output> [hidden_states, attentions]
        Note that: in doc, it says <last_hidden_state> is better semantic summery than <pooler_output>.
        thus, we use <last_hidden_state>.
        """
        super().__init__()
        # self.model_name_or_path = model_name_or_path
        encoder_type = EncoderType.from_string(encoder_type) if isinstance(encoder_type, str) else encoder_type
        if encoder_type not in list(EncoderType):
            raise ValueError(f"{encoder_type} encoder_type must be in {list(EncoderType)}")
        self.encoder_type = encoder_type
        self.max_seq_length = max_seq_length
        # self.bert = AutoModel.from_pretrained(model_name_or_path)

        """
        bert最后添加了一层pool层(pooler): BertPooler(
                                (dense): Linear(in_features=768, out_features=768, bias=True)
                                (activation): Tanh()
                              )
        分布式训练时，如果未使用POOLER的输出作为特征表示，而是使用了hidden_states，last_hidden_state，则梯度反向传播时，pooler层没有计算梯度
        报错 Parameters which did not receive grad for rank 0: pooler.dense.bias, pooler.dense.weight（）
        提示：https://github.com/allenai/allennlp/discussions/5509
        bert网络结构：https://towardsdatascience.com/deep-dive-into-the-code-of-bert-model-9f618472353e
        只需要在你正常的分布式命令前加入​​TORCH_DISTRIBUTED_DEBUG=DETAIL​​即可查看哪些参数初始化了，但是并没有在模型的foward过程中使用，因此没有梯度无法反传参数更新。：
        TORCH_DISTRIBUTED_DEBUG=DETAIL bash tools/dist_train.sh config/xxx.py 1
        """
        if self.encoder_type == EncoderType.POOLER:
            self.bert = BertModel.from_pretrained(model_name_or_path)
        else:
            self.bert = BertModel.from_pretrained(model_name_or_path, add_pooling_layer=False)
            # self.bert = BertModel.from_pretrained(model_name_or_path)

        # self.device = torch.device(device)
        # logger.debug("Use device: {}".format(self.device))
        self.bert.to(device)
        self.results = {}  # Save training process evaluation result

    def __str__(self):
        return f"<SentenceModel: {self.model_name_or_path}, encoder_type: {self.encoder_type}, " \
               f"max_seq_length: {self.max_seq_length}, emb_dim: {self.get_sentence_embedding_dimension()}>"

    def get_sentence_embedding_dimension(self):
        """
        Get the dimension of the sentence embeddings.

        Returns
        -------
        int or None
            The dimension of the sentence embeddings, or None if it cannot be determined.
        """
        # Use getattr to safely access the out_features attribute of the pooler's dense layer
        return getattr(self.bert.pooler.dense, "out_features", None)

    def get_sentence_embeddings(self, input_ids, attention_mask, token_type_ids=None):
        """
        Returns the model output by encoder_type as embeddings.

        Utility function for self.bert() method.
        """
        model_output = self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True)

        if self.encoder_type == EncoderType.FIRST_LAST_AVG:
            # Get the first and last hidden states, and average them to get the embeddings
            # hidden_states have 13 list, second is hidden_state
            first = model_output.hidden_states[1]
            last = model_output.hidden_states[-1]
            seq_length = first.size(1)  # Sequence length

            first_avg = torch.avg_pool1d(first.transpose(1, 2), kernel_size=seq_length).squeeze(-1)  # [batch, hid_size]
            last_avg = torch.avg_pool1d(last.transpose(1, 2), kernel_size=seq_length).squeeze(-1)  # [batch, hid_size]
            final_encoding = torch.avg_pool1d(
                torch.cat([first_avg.unsqueeze(1), last_avg.unsqueeze(1)], dim=1).transpose(1, 2),
                kernel_size=2).squeeze(-1)
            return final_encoding
        #
        if self.encoder_type == EncoderType.LAST_AVG:
            sequence_output = model_output.last_hidden_state  # [batch_size, max_len, hidden_size]
            seq_length = sequence_output.size(1)
            final_encoding = torch.avg_pool1d(sequence_output.transpose(1, 2), kernel_size=seq_length).squeeze(-1)
            return final_encoding

        if self.encoder_type == EncoderType.CLS:
            sequence_output = model_output.last_hidden_state
            return sequence_output[:, 0]  # [batch, hid_size]

        if self.encoder_type == EncoderType.POOLER:
            return model_output.pooler_output  # [batch, hid_size]

        if self.encoder_type == EncoderType.MEAN:
            """
            Mean Pooling - Take attention mask into account for correct averaging
            """
            token_embeddings = model_output.last_hidden_state  # Contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            final_encoding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9)
            return final_encoding  # [batch, hid_size]

    def calc_loss(self, y_true, y_pred):
        """
        矩阵计算batch内的cos loss
        """
        # 1. 取出真实的标签
        y_true = y_true[::2]  # tensor([1, 0, 1]) 真实的标签
        # 2. 对输出的句子向量进行l2归一化   后面只需要对应为相乘  就可以得到cos值了
        norms = (y_pred ** 2).sum(axis=1, keepdims=True) ** 0.5
        y_pred = y_pred / norms
        # 3. 奇偶向量相乘, 相似度矩阵除以温度系数0.05(等于*20)
        y_pred = torch.sum(y_pred[::2] * y_pred[1::2], dim=1) * 20
        # 4. 取出负例-正例的差值
        y_pred = y_pred[:, None] - y_pred[None, :]  # 这里是算出所有位置 两两之间余弦的差值
        # 矩阵中的第i行j列  表示的是第i个余弦值-第j个余弦值
        y_true = y_true[:, None] < y_true[None, :]  # 取出负例-正例的差值
        y_true = y_true.float()
        y_pred = y_pred - (1 - y_true) * 1e12
        y_pred = y_pred.view(-1)
        # 这里加0是因为e^0 = 1相当于在log中加了1
        y_pred = torch.cat((torch.tensor([0]).float().to(device), y_pred), dim=0)
        # y_pred = torch.cat((torch.tensor([0]).float().to(), y_pred), dim=0)
        return torch.logsumexp(y_pred, dim=0)

    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        output_embeddings = self.get_sentence_embeddings(input_ids, attention_mask, token_type_ids)
        loss = self.calc_loss(labels, output_embeddings)
        return loss


class CosentModel():
    def __init__(
            self,
            model_name_or_path: str = "hfl/chinese-macbert-base",
            encoder_type: str = "FIRST_LAST_AVG",
            max_seq_length: int = 128,
    ):
        """
        Initializes a CoSENT Model.

        Args:
            model_name_or_path: Default Transformer model name or path to a directory containing Transformer model file (pytorch_nodel.bin).
            encoder_type: Enum of type EncoderType.
            max_seq_length: The maximum total input sequence length after tokenization.
            device: The device on which the model is allocated.
        """
        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length
        # self.encoder_type=encoder_type
        self.model = EncodeModel(model_name_or_path, encoder_type, max_seq_length)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        # self.tokenizer =self.model.tokenizer

    # def __str__(self):
    #     return f"<CoSENTModel: {self.model_name_or_path}, encoder_type: {self.encoder_type}, " \
    #            f"max_seq_length: {self.max_seq_length}>"

    def train_model(
            self,
            train_file: str = None,
            output_dir: str = None,
            eval_file: str = None,
            verbose: bool = True,
            batch_size: int = 32,
            num_epochs: int = 1,
            weight_decay: float = 0.01,
            seed: int = 42,
            warmup_ratio: float = 0.05,
            lr: float = 2e-5,
            eps: float = 1e-6,
            gradient_accumulation_steps: int = 1,
            max_grad_norm: float = 1.0,
            max_steps: int = -1,
            use_hf_dataset: bool = False,
            hf_dataset_name: str = "STS-B",
            save_model_every_epoch: bool = True,
            bf16: bool = False,
            data_parallel: bool = False,
            local_rank: int = -1
    ):
        """
        Trains the model on 'train_file'

        Args:
            train_file: Path to text file containing the text to _train the language model on.
            output_dir: The directory where model files will be saved. If not given, self.args.output_dir will be used.
            eval_file (optional): Path to eval file containing the text to _evaluate the language model on.
            verbose (optional): Print logger or not.
            batch_size (optional): Batch size for training.
            num_epochs (optional): Number of epochs for training.
            weight_decay (optional): Weight decay for optimization.
            seed (optional): Seed for initialization.
            warmup_ratio (optional): Warmup ratio for learning rate.
            lr (optional): Learning rate.
            eps (optional): Adam epsilon.
            gradient_accumulation_steps (optional): Number of updates steps to accumulate before performing a backward/update pass.
            max_grad_norm (optional): Max gradient norm.
            max_steps (optional): If > 0: set total number of training steps to perform. Override num_epochs.
            use_hf_dataset (optional): Whether to use the HF dataset.
            hf_dataset_name (optional): Name of the dataset to use for the HuggingFace datasets.
            save_model_every_epoch (optional): Save model checkpoint every epoch.
            bf16 (optional): Use bfloat16 amp training.
            data_parallel (optional): Use multi-gpu data parallel training.
        Returns:
            global_step: Number of global steps trained
            training_details: full training progress scores
        """
        if use_hf_dataset and hf_dataset_name:
            logger.info(
                f"Train_file will be ignored when use_hf_dataset is True, load HF dataset: {hf_dataset_name}")
            train_dataset = HFCosentTrainDataset(self.tokenizer, hf_dataset_name, max_len=self.max_seq_length)
            eval_dataset = HFTextMatchingTestDataset(self.tokenizer, hf_dataset_name, max_len=self.max_seq_length)
        elif train_file is not None:
            logger.info(
                f"Hf_dataset_name: {hf_dataset_name} will be ignored when use_hf_dataset is False, load train_file: {train_file}")
            train_dataset = CosentTrainDataset(self.tokenizer, load_cosent_train_data(train_file), self.max_seq_length)
            eval_dataset = TextMatchingTestDataset(self.tokenizer, load_text_matching_test_data(eval_file),
                                                   self.max_seq_length)
        else:
            raise ValueError("Error, train_file|use_hf_dataset must be specified")

        global_step, training_details = self.train(
            train_dataset,
            output_dir,
            eval_dataset=eval_dataset,
            verbose=verbose,
            batch_size=batch_size,
            num_epochs=num_epochs,
            weight_decay=weight_decay,
            seed=seed,
            warmup_ratio=warmup_ratio,
            lr=lr,
            eps=eps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=max_grad_norm,
            max_steps=max_steps,
            save_model_every_epoch=save_model_every_epoch,
            bf16=bf16,
            data_parallel=data_parallel,
            local_rank=local_rank
        )
        logger.info(f" Training model done. Saved to {output_dir}.")

        return global_step, training_details

    def train(
            self,
            train_dataset: Dataset,
            output_dir: str,
            eval_dataset: Dataset = None,
            verbose: bool = True,
            batch_size: int = 8,
            num_epochs: int = 1,
            weight_decay: float = 0.01,
            seed: int = 42,
            warmup_ratio: float = 0.05,
            lr: float = 2e-5,
            eps: float = 1e-6,
            gradient_accumulation_steps: int = 1,
            max_grad_norm: float = 1.0,
            max_steps: int = -1,
            save_model_every_epoch: bool = True,
            bf16: bool = False,
            data_parallel: bool = False,
            local_rank: int = -1
    ):
        """
        Trains the model on train_dataset.

        Utility function to be used by the train_model() method. Not intended to be used directly.
        """
        if data_parallel:
            global device
            device = torch.device("cuda:{}".format(local_rank))
        os.makedirs(output_dir, exist_ok=True)
        # self.device=torch.device(f'cuda:{local_rank}')
        logger.debug("Use device: {}".format(device))

        # self.bert.to(self.device)
        # self.model.bert.to(device)
        self.model.bert.to(device)
        set_seed(seed)

        if data_parallel:
            # dp数据分布
            # self.bert = nn.DataParallel(self.bert)
            # ddp数据分布
            self.model.bert = torch.nn.parallel.DistributedDataParallel(self.model.bert, device_ids=[local_rank],
                                                                        output_device=local_rank)
        if data_parallel:
            # ddp数据分布
            # train_sampler =DistributedSampler(train_dataset,shuffle=False)  #是否打乱数据，默认打乱
            train_sampler = DistributedSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        else:
            train_dataloader = DataLoader(train_dataset,
                                          batch_size=batch_size, )  # keep the order of the data, not shuffle

        eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)
        total_steps = len(train_dataloader) * num_epochs
        param_optimizer = list(self.model.bert.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        warmup_steps = math.ceil(total_steps * warmup_ratio)  # by default 10% of _train data for warm-up
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=eps, correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=total_steps)
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Batch size = {batch_size}")
        logger.info(f"  Num steps = {total_steps}")
        logger.info(f"  Warmup-steps: {warmup_steps}")

        logger.info("  Training started")
        global_step = 0
        self.model.bert.zero_grad()
        epoch_number = 0
        best_eval_metric = 0
        steps_trained_in_current_epoch = 0
        epochs_trained = 0

        if self.model_name_or_path and os.path.exists(self.model_name_or_path):
            try:
                # set global_step to global_step of last saved checkpoint from model path
                checkpoint_suffix = self.model_name_or_path.split("/")[-1].split("-")
                if len(checkpoint_suffix) > 2:
                    checkpoint_suffix = checkpoint_suffix[1]
                else:
                    checkpoint_suffix = checkpoint_suffix[-1]
                global_step = int(checkpoint_suffix)
                epochs_trained = global_step // (len(train_dataloader) // gradient_accumulation_steps)
                steps_trained_in_current_epoch = global_step % (len(train_dataloader) // gradient_accumulation_steps)
                logger.info("   Continuing training from checkpoint, will skip to saved global_step")
                logger.info("   Continuing training from epoch %d" % epochs_trained)
                logger.info("   Continuing training from global step %d" % global_step)
                logger.info("   Will skip the first %d steps in the current epoch" % steps_trained_in_current_epoch)
            except ValueError:
                logger.info("   Starting fine-tuning.")

        training_progress_scores = {
            "global_step": [],
            "train_loss": [],
            "eval_spearman": [],
            "eval_pearson": [],
        }
        for current_epoch in trange(int(num_epochs), desc="Epoch", disable=False, mininterval=0):
            self.model.bert.train()
            if data_parallel:
                """
                 在分布式模式下，需要在每个 epoch 开始时调用 set_epoch() 方法，然后再创建 DataLoader 迭代器，以使 shuffle 操作能够在多个 epoch 中正常工作。 否则，dataloader迭代器产生的数据将始终使用相同的顺序。
                """
                train_sampler.set_epoch(current_epoch)  #
            current_loss = 0
            if epochs_trained > 0:
                epochs_trained -= 1
                continue
            batch_iterator = tqdm(train_dataloader,
                                  desc=f"Running Epoch {epoch_number + 1} of {num_epochs}",
                                  disable=False,
                                  mininterval=0)
            for step, batch in enumerate(batch_iterator):
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue
                inputs, labels = batch
                labels = labels.to(device)
                # inputs        [batch, 1, seq_len] -> [batch, seq_len]
                input_ids = inputs.get('input_ids').squeeze(1).to(device)
                attention_mask = inputs.get('attention_mask').squeeze(1).to(device)
                token_type_ids = inputs.get('token_type_ids', None)
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.squeeze(1).to(device)

                if bf16:
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        # output_embeddings = self.get_sentence_embeddings(input_ids, attention_mask, token_type_ids)
                        # loss = self.calc_loss(labels, output_embeddings)
                        loss = self.model(input_ids, attention_mask, token_type_ids, labels)
                else:
                    # output_embeddings = self.get_sentence_embeddings(input_ids, attention_mask, token_type_ids)
                    # loss = self.calc_loss(labels, output_embeddings)
                    # loss,model_output,output_embeddings = self.model(input_ids, attention_mask, token_type_ids, labels,device)
                    loss = self.model(input_ids, attention_mask, token_type_ids, labels)

                current_loss = loss.item()
                if verbose:
                    batch_iterator.set_description(
                        f"Epoch: {epoch_number + 1}/{num_epochs}, Batch:{step}/{len(train_dataloader)}, Loss: {current_loss:9.4f}")

                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                loss.backward()
                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.bert.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    optimizer.zero_grad()
                    global_step += 1

            epoch_number += 1
            output_dir_current = os.path.join(output_dir, "checkpoint-{}-epoch-{}".format(global_step, epoch_number))
            if local_rank == 0 or not data_parallel:
                # results = self.eval_model(eval_dataset, output_dir_current, verbose=verbose, batch_size=batch_size)
                results = self.evaluate(eval_dataloader, output_dir_current, batch_size=batch_size)
                if save_model_every_epoch:
                    self.save_model(output_dir_current, model=self.model.bert, results=results)
                training_progress_scores["global_step"].append(global_step)
                training_progress_scores["train_loss"].append(current_loss)
                for key in results:
                    training_progress_scores[key].append(results[key])
                report = pd.DataFrame(training_progress_scores)
                report.to_csv(os.path.join(output_dir, "training_progress_scores.csv"), index=False)

                eval_spearman = results["eval_spearman"]
                if eval_spearman > best_eval_metric:
                    best_eval_metric = eval_spearman
                    logger.info(f"Save new best model, best_eval_metric: {best_eval_metric}")
                    self.save_model(output_dir, model=self.model.bert, results=results)

            if 0 < max_steps < global_step:
                return global_step, training_progress_scores

        return global_step, training_progress_scores

    def evaluate(self, eval_dataloader, output_dir: str = None, batch_size: int = 16):
        """
        Evaluates the model on eval_dataset.

        Utility function to be used by the eval_model() method. Not intended to be used directly.
        """
        results = {}

        batch_labels = []
        batch_preds = []
        for batch in tqdm(eval_dataloader, disable=False, desc="Running Evaluation"):
            source, target, labels = batch
            labels = labels.to(device)
            batch_labels.extend(labels.cpu().numpy())
            # source        [batch, 1, seq_len] -> [batch, seq_len]
            source_input_ids = source.get('input_ids').squeeze(1).to(device)
            source_attention_mask = source.get('attention_mask').squeeze(1).to(device)
            source_token_type_ids = source.get('token_type_ids', None)
            if source_token_type_ids is not None:
                source_token_type_ids = source_token_type_ids.squeeze(1).to(device)

            # target        [batch, 1, seq_len] -> [batch, seq_len]
            target_input_ids = target.get('input_ids').squeeze(1).to(device)
            target_attention_mask = target.get('attention_mask').squeeze(1).to(device)
            target_token_type_ids = target.get('token_type_ids', None)
            if target_token_type_ids is not None:
                target_token_type_ids = target_token_type_ids.squeeze(1).to(device)

            with torch.no_grad():
                # source_embeddings = self.get_sentence_embeddings(source_input_ids, source_attention_mask,
                #                                                  source_token_type_ids)
                # target_embeddings = self.get_sentence_embeddings(target_input_ids, target_attention_mask,
                #                                                  target_token_type_ids)
                source_embeddings = self.model.get_sentence_embeddings(source_input_ids, source_attention_mask,
                                                                       source_token_type_ids)
                target_embeddings = self.model.get_sentence_embeddings(target_input_ids, target_attention_mask,
                                                                       target_token_type_ids)
                preds = torch.cosine_similarity(source_embeddings, target_embeddings)
            batch_preds.extend(preds.cpu().numpy())

        spearman = compute_spearmanr(batch_labels, batch_preds)
        pearson = compute_pearsonr(batch_labels, batch_preds)
        logger.debug(f"labels: {batch_labels[:10]}")
        logger.debug(f"preds:  {batch_preds[:10]}")
        logger.debug(f"pearson: {pearson}, spearman: {spearman}")

        results["eval_spearman"] = spearman
        results["eval_pearson"] = pearson
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "eval_results.txt"), "w") as writer:
                for key in sorted(results.keys()):
                    writer.write("{} = {}\n".format(key, str(results[key])))

        return results

    def save_model(self, output_dir, model, results=None):
        """
        Saves the model to output_dir.
        :param output_dir:
        :param model:
        :param results:
        :return:
        """
        logger.debug(f"Saving model checkpoint to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        if results:
            output_eval_file = os.path.join(output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                for key in sorted(results.keys()):
                    writer.write("{} = {}\n".format(key, str(results[key])))