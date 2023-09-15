# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

import numpy as np
from itertools import chain

from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef as _matthews_corrcoef
from scipy.stats import pearsonr, spearmanr

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.logging.meters import safe_round

def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def matthews_corrcoef(preds, labels):
    # make it consistent with other metrics taking (preds, labels) as input
    mcc = _matthews_corrcoef(labels, preds)
    return mcc


@register_criterion('sentence_prediction')
class SentencePredictionCriterion(FairseqCriterion):

    def __init__(self, task, classification_head_name, regression_target):
        super().__init__(task)
        self.classification_head_name = classification_head_name
        self.regression_target = regression_target

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--classification-head-name',
                            default='sentence_classification_head',
                            help='name of the classification head to use')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert (
            hasattr(model, 'classification_heads')
            and self.classification_head_name in model.classification_heads
        ), 'model must provide sentence classification head for --criterion=sentence_prediction'

        logits, _ = model(
            **sample['net_input'],
            features_only=True,
            classification_head_name=self.classification_head_name,
        )
        targets = model.get_targets(sample, [logits]).view(-1)
        sample_size = targets.numel()

        if not self.regression_target:
            lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
            loss = F.nll_loss(lprobs, targets, reduction='sum')
        else:
            logits = logits.view(-1).float()
            targets = targets.float()
            loss = F.mse_loss(logits, targets, reduction='sum')

        logging_output = {
            'loss': loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample_size,
            'sample_size': sample_size,
        }
        if not self.regression_target:
            preds = logits.argmax(dim=1)
            logging_output['ncorrect'] = (preds == targets).sum()
            logging_output['false'] = (preds != targets).sum()
            logging_output['tp'] = ((preds == 1) & (targets == 1)).sum()
            logging_output['tn'] = ((preds == 0) & (targets == 0)).sum()
            logging_output['fp'] = ((preds == 1) & (targets == 0)).sum()
            logging_output['fn'] = ((preds == 0) & (targets == 1)).sum()
        else:
            logging_output["pred"] = logits.detach().cpu().tolist()
            logging_output["targ"] = targets.detach().cpu().tolist()

            logging_output["report_mcc"] = True
            logging_output["report_acc_and_f1"] = True
            logging_output["report_pearson_and_spearman"] = True
        
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        if sample_size != ntokens:
            metrics.log_scalar('nll_loss', loss_sum / ntokens / math.log(2), ntokens, round=3)

        if len(logging_outputs) > 0 and 'ncorrect' in logging_outputs[0]:
            ncorrect = sum(log.get('ncorrect', 0) for log in logging_outputs)
            false = sum(log.get('false', 0) for log in logging_outputs)
            tp = sum(log.get('tp', 0) for log in logging_outputs)
            tn = sum(log.get('tn', 0) for log in logging_outputs)
            fp = sum(log.get('fp', 0) for log in logging_outputs)
            fn = sum(log.get('fn', 0) for log in logging_outputs)
            metrics.log_scalar('accuracy', 100.0 * ncorrect / nsentences, nsentences, round=1)
            metrics.log_scalar('tp', 1. * tp, 1., round=1, sum_meter=True)
            metrics.log_scalar('tn', 1. * tn, 1., round=1, sum_meter=True)
            metrics.log_scalar('fp', 1. * fp, 1., round=1, sum_meter=True)
            metrics.log_scalar('fn', 1. * fn, 1., round=1, sum_meter=True)
            metrics.log_scalar('false', 1. * false, 1., round=1, sum_meter=True)
        
        # Metrics used by GLUE
        pred = np.array(
            list(chain.from_iterable(log.get("pred", []) for log in logging_outputs))
        )
        targ = np.array(
            list(chain.from_iterable(log.get("targ", []) for log in logging_outputs))
        )
        
        if len(pred):
            metrics.log_concat_tensor("pred", torch.from_numpy(pred), dim=0)
            metrics.log_concat_tensor("targ", torch.from_numpy(targ), dim=0)
            if any("report_pearson_and_spearman" in log for log in logging_outputs):
                metrics.log_derived(
                    "pearson_and_spearman",
                    lambda meters: safe_round(
                        pearson_and_spearman(
                            meters["pred"].tensor.numpy(),
                            meters["targ"].tensor.numpy(),
                        )["corr"]
                        * 100,
                        1,
                    ),
                )
                metrics.log_derived(
                    "pearson",
                    lambda meters: safe_round(
                        pearson_and_spearman(
                            meters["pred"].tensor.numpy(),
                            meters["targ"].tensor.numpy(),
                        )["pearson"]
                        * 100,
                        1,
                    ),
                )
                metrics.log_derived(
                    "spearman",
                    lambda meters: safe_round(
                        pearson_and_spearman(
                            meters["pred"].tensor.numpy(),
                            meters["targ"].tensor.numpy(),
                        )["spearmanr"]
                        * 100,
                        1,
                    ),
                )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
