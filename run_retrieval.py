# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""BERT finetuning runner."""

import argparse
import json
import logging
import os
import random
from io import open

import numpy
from time import gmtime, strftime
from timeit import default_timer as timer

from tensorboardX import SummaryWriter
from tqdm import tqdm
from bisect import bisect

import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.utils.data import DataLoader

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

from multimodal_bert.datasets import COCORetreivalDataset
from multimodal_bert.datasets._image_features_reader import ImageFeaturesH5Reader
import pdb

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    # Data files for VQA task.
    parser.add_argument("--features_h5path", default="/coc/pskynet2/jlu347/multi-modal-bert/data/coco/coco_trainval.h5")
    parser.add_argument(
        "--train_file",
        default="data/cocoRetreival/coco/dataset.json",
        type=str,
        # required=True,
        help="The input train corpus.",
    )
    parser.add_argument(
        "--bert_model",
        default="bert-base-uncased",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    )

    parser.add_argument(
        "--pretrained_weight",
        default="bert-base-uncased",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    )

    parser.add_argument(
        "--output_dir",
        default="save",
        type=str,
        # required=True,
        help="The output directory where the model checkpoints will be written.",
    )

    parser.add_argument(
        "--config_file",
        default="config/bert_config.json",
        type=str,
        # required=True,
        help="The config file which specified the model details.",
    )
    ## Other parameters
    parser.add_argument(
        "--max_seq_length",
        default=30,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. \n"
        "Sequences longer than this will be truncated, and sequences shorter \n"
        "than this will be padded.",
    )

    parser.add_argument("--use_location", action="store_true", help="whether use location.")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument(
        "--train_batch_size", default=128, type=int, help="Total batch size for training."
    )
    parser.add_argument(
        "--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=15,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.01,
        type=float,
        help="Proportion of training to perform linear learning rate warmup for. "
        "E.g., 0.1 = 10%% of training.",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Whether not to use CUDA when available"
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Whether to lower case the input text. True for uncased models, False for cased models.",
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus"
    )
    
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumualte before performing a backward/update pass.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit float precision instead of 32-bit",
    )
    parser.add_argument(
        "--loss_scale",
        type=float,
        default=0,
        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
        "0 (default value): dynamic loss scaling.\n"
        "Positive power of 2: static loss scaling value.\n",
    )
    parser.add_argument(
        "--num_workers", type=int, default=20, help="Number of workers in the dataloader."
    )
    parser.add_argument(
        "--from_pretrained", action="store_true", help="Wheter the tensor is from pretrained."
    )
    parser.add_argument(
        "--save_name",
        default='',
        type=str,
        help="save name for training.",
    )
    parser.add_argument(
        "--baseline", action="store_true", help="Wheter to use the baseline model (single bert)."
    )
    parser.add_argument(
        "--split", default='train', type=str, help="train or trainval."
    )
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')

    parser.add_argument('--max_violation', action='store_true',
                        help='Use max instead of sum in the rank loss.')

    parser.add_argument('--measure', default='cosine',
                        help='Similarity measure used (cosine|order)')

    args = parser.parse_args()

    if args.baseline:
        from pytorch_pretrained_bert.modeling import BertConfig
        from multimodal_bert.bert import MultiModalBertForImageCaptionRetrieval
    else:
        from multimodal_bert.multi_modal_bert import MultiModalBertForImageCaptionRetrieval, BertConfig

    print(args)
    if args.save_name is not '':
        timeStamp = args.save_name
    else:
        timeStamp = strftime("%d-%b-%y-%X-%a", gmtime())
        timeStamp += "_{:0>6d}".format(random.randint(0, 10e6))
    
    savePath = os.path.join(args.output_dir, timeStamp)

    if not os.path.exists(savePath):
        os.makedirs(savePath)
    
    config = BertConfig.from_json_file(args.config_file)
    # save all the hidden parameters. 
    with open(os.path.join(savePath, 'command.txt'), 'w') as f:
        print(args, file=f)  # Python 3.x
        print('\n', file=f)
        print(config, file=f)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend="nccl")
    logger.info(
        "device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            device, n_gpu, bool(args.local_rank != -1), args.fp16
        )
    )

    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.gradient_accumulation_steps
            )
        )

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # if n_gpu > 0:
    #     torch.cuda.manual_seed_all(args.seed)

    if not args.do_train:
        raise ValueError(
            "Training is currently the only implemented execution option. Please set `do_train`."
        )

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
    #     raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # train_examples = None
    num_train_optimization_steps = None
    if args.do_train:

        viz = TBlogger("logs/" + timeStamp)

        print("Loading Train Dataset", args.train_file)

        tokenizer = BertTokenizer.from_pretrained(
            args.bert_model, do_lower_case=args.do_lower_case
        )
        image_features_reader_train = ImageFeaturesH5Reader(args.features_h5path, True)
        image_features_reader_val = ImageFeaturesH5Reader(args.features_h5path, True)

        train_dset = COCORetreivalDataset(
            "train", args.train_file, image_features_reader_train, tokenizer)
        eval_dset = COCORetreivalDataset("val", args.train_file, image_features_reader_val, tokenizer)

        # dictionary = BertDictionary(args)        
        # train_dset = BertFeatureDataset('train', dictionary, dataroot='data/VQA')
        # eval_dset = BertFeatureDataset('val', dictionary, dataroot='data/VQA')

        num_train_optimization_steps = (
            int(len(train_dset) / args.train_batch_size / args.gradient_accumulation_steps)
            * args.num_train_epochs
        )
        if args.local_rank != -1:
            num_train_optimization_steps = (
                num_train_optimization_steps // torch.distributed.get_world_size()
            )

    # num_labels = 3000
    if args.from_pretrained:
        model = MultiModalBertForImageCaptionRetrieval.from_pretrained(
            args.pretrained_weight, config
        )
    else:
        model = MultiModalBertForImageCaptionRetrieval.from_pretrained(
            args.bert_model, config
        )

    if args.fp16:
        model.half()
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )
        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    model.cuda()
    # pdb.set_trace()
    # Prepare optimizer
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    if not args.from_pretrained:
        param_optimizer = list(model.named_parameters())
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
    else:
        bert_weight_name = json.load(open("config/bert_weight_name.json", "r"))
        optimizer_grouped_parameters = []
        for key, value in dict(model.named_parameters()).items():
            if value.requires_grad:
                if key[12:] in bert_weight_name:
                    lr = args.learning_rate
                else:
                    lr = args.learning_rate

                if any(nd in key for nd in no_decay):
                    optimizer_grouped_parameters += [
                        {"params": [value], "lr": lr}
                    ]

                if not any(nd in key for nd in no_decay):
                    optimizer_grouped_parameters += [
                        {"params": [value], "lr": lr}
                    ]

    # set different parameters for vision branch and lanugage branch.
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )

        optimizer = FusedAdam(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            bias_correction=False,
            max_grad_norm=1.0,
        )
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
            warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,
                                                 t_total=num_train_optimization_steps)
    else:
        if args.from_pretrained:
            # optimizer = BertAdam(
            #     optimizer_grouped_parameters,
            #     warmup=args.warmup_proportion,
            #     t_total=num_train_optimization_steps,
            # )
            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=args.learning_rate,
                                 warmup=args.warmup_proportion,
                                 t_total=num_train_optimization_steps)
        else:
            # optimizer = BertAdam(
            #     optimizer_grouped_parameters,
            #     lr=args.learning_rate,
            #     warmup=args.warmup_proportion,
            #     t_total=num_train_optimization_steps,
            # )
            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=args.learning_rate,
                                 warmup=args.warmup_proportion,
                                 t_total=num_train_optimization_steps)

    # lr_lambda = lambda x: lr_lambda_update(x)
    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    loss_fun = ContrastiveLoss(margin=args.margin,
                                         measure=args.measure,
                                         max_violation=args.max_violation)
    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dset))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        train_dataloader = DataLoader(
            train_dset,
            shuffle=True,
            batch_size=args.train_batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        eval_dataloader = DataLoader(
            eval_dset,
            shuffle=False,
            batch_size=args.train_batch_size,
            num_workers=100,
            pin_memory=True,
        )

        startIterID = 0
        global_step = 0
        masked_loss_v_tmp = 0
        masked_loss_t_tmp = 0
        next_sentence_loss_tmp = 0
        loss_tmp = 0
        start_t = timer()

        model.train()
        # t1 = timer()
        for epochId in tqdm(range(args.num_train_epochs), desc="Epoch"):
            total_loss = 0
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            train_score = 0
            optimizer.zero_grad()

            # iter_dataloader = iter(train_dataloader)
            for step, batch in enumerate(train_dataloader):
                iterId = startIterID + step + (epochId * len(train_dataloader))
                batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)

                features, spatials, image_mask, caption, input_mask, segment_ids = batch
                
                t_feature, v_feature = model(caption, features, spatials, segment_ids, input_mask, image_mask)

                loss = loss_fun(v_feature, t_feature)

                # nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                total_loss += loss.item()

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                # print(loss)
                # print(tr_loss)
                viz.linePlot(iterId, loss.item(), "loss", "train")
                # viz.linePlot(iterId, optimizer.get_lr()[0], 'learning_rate', 'train')

                loss_tmp += loss.item()

                nb_tr_examples += caption.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(
                            global_step / num_train_optimization_steps, args.warmup_proportion
                        )
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = lr_this_step
        
                    nn.utils.clip_grad_norm_(model.parameters(), 0.25)

                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                # lr_scheduler.step(iterId)
                if step % 20 == 0 and step != 0:
                    loss_tmp = loss_tmp / 20.0

                    end_t = timer()
                    timeStamp = strftime("%a %d %b %y %X", gmtime())

                    Ep = epochId + nb_tr_steps / float(len(train_dataloader))
                    printFormat = "[%s][Ep: %.2f][Iter: %d][Time: %5.2fs][Loss: %.5g]"

                    printInfo = [
                        timeStamp,
                        Ep,
                        iterId,
                        end_t - start_t,
                        loss_tmp,
                    ]

                    start_t = end_t
                    print(printFormat % tuple(printInfo))

                    loss_tmp = 0

            model.train(False)
            eval_loss = evaluate(args, model, eval_dataloader)
            model.train(True)

            total_loss = total_loss / len(train_dataloader)
            logger.info("epoch %d" % (epochId))
            logger.info("\ttrain_loss: %.2f" % (total_loss))
            logger.info("\teval_loss: %.2f" % (eval_loss))
            
            # Save a trained model
            logger.info("** ** * Saving fine - tuned model ** ** * ")
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Only save the model it-self

            if not os.path.exists(savePath):
                os.makedirs(savePath)
            output_model_file = os.path.join(savePath, "pytorch_model_" + str(epochId) + ".bin")
            if args.do_train:
                torch.save(model_to_save.state_dict(), output_model_file)


class TBlogger:
    def __init__(self, log_dir):
        print("logging file at: " + log_dir)
        self.logger = SummaryWriter(log_dir=log_dir)

    def linePlot(self, step, val, split, key, xlabel="None"):
        self.logger.add_scalar(split + "/" + key, val, step)

def evaluate(args, model, dataloader, loss_fun):
    score = 0
    total_loss = 0
    num_data = 0
    for batch in tqdm(iter(dataloader)):
        batch = tuple(t.cuda() for t in batch)
        features, spatials, image_mask, caption, input_mask, segment_ids = batch
        with torch.no_grad():
            t_feature, v_feature = model(caption, features, spatials, segment_ids, input_mask, image_mask)

        total_loss += loss_fun(v_feature, t_feature)

        # total_loss += loss_fun(v_feature, t_feature).sum().item()
        # v_feature = v_feature.cpu().numpy()
        # t_feature = t_feature.cpu().numpy()

        # r, rt = i2t(v_feature, t_feature, measure=args.measure, return_ranks=True)
        # ri, rti = t2i(v_feature, t_feature, measure=args.measure, return_ranks=True)

        # ar = (r[0] + r[1] + r[2]) / 3
        # ari = (ri[0] + ri[1] + ri[2]) / 3
        # rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        # print("rsum: %.1f" % rsum)
        # print("Average i2t Recall: %.1f" % ar)
        # print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
        # print("Average t2i Recall: %.1f" % ari)
        # print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)

        # batch_score = compute_score_with_logits(pred, target.cuda()).sum()
        # score += batch_score.item()
        # upper_bound += (target.max(1)[0]).sum()
    return total_loss / float(dataloader)

class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        if measure == 'order':
            self.sim = order_sim
        else:
            self.sim = cosine_sim

        self.max_violation = max_violation

    def forward(self, im, s):
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = mask
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()

def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())

def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score

def lr_lambda_update(i_iter):
    warmup_iterations = 1000
    warmup_factor = 0.2
    lr_ratio = 0.1
    lr_steps = [15000, 18000, 20000, 21000]
    if i_iter <= warmup_iterations:
        alpha = float(i_iter) / float(warmup_iterations)
        return warmup_factor * (1.0 - alpha) + alpha
    else:
        idx = bisect([], i_iter)
        return pow(lr_ratio, idx)

def i2t(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = images.shape[0] / 5
    index_list = []

    ranks = numpy.zeros(npts)
    top1 = numpy.zeros(npts)
    for index in range(npts):

        # Get query image
        im = images[5 * index].reshape(1, images.shape[1])

        # Compute scores
        if measure == 'order':
            bs = 100
            if index % bs == 0:
                mx = min(images.shape[0], 5 * (index + bs))
                im2 = images[5 * index:mx:5]
                d2 = order_sim(torch.Tensor(im2).cuda(),
                               torch.Tensor(captions).cuda())
                d2 = d2.cpu().numpy()
            d = d2[index % bs]
        else:
            d = numpy.dot(im, captions.T).flatten()
        inds = numpy.argsort(d)[::-1]
        index_list.append(inds[0])

        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = numpy.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = images.shape[0] / 5
    ims = numpy.array([images[i] for i in range(0, len(images), 5)])

    ranks = numpy.zeros(5 * npts)
    top1 = numpy.zeros(5 * npts)
    for index in range(npts):

        # Get query captions
        queries = captions[5 * index:5 * index + 5]

        # Compute scores
        if measure == 'order':
            bs = 100
            if 5 * index % bs == 0:
                mx = min(captions.shape[0], 5 * index + bs)
                q2 = captions[5 * index:mx]
                d2 = order_sim(torch.Tensor(ims).cuda(),
                               torch.Tensor(q2).cuda())
                d2 = d2.cpu().numpy()

            d = d2[:, (5 * index) % bs:(5 * index) % bs + 5].T
        else:
            d = numpy.dot(queries, ims.T)
        inds = numpy.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = numpy.argsort(d[i])[::-1]
            ranks[5 * index + i] = numpy.where(inds[i] == index)[0][0]
            top1[5 * index + i] = inds[i][0]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)

if __name__ == "__main__":

    main()
