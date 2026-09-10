"""Comprehensive Trainer for NSSG-DimNet and Baselines (Phase 6).

Supports:
- Separate parameter groups (pretrained backbone vs new architectural modules)
- AdamW optimizer with configurable weight decay
- Linear warmup and learning rate decay scheduling
- Gradient accumulation and gradient clipping
- Safe epoch-level accumulation of continuous predictions and targets for valid CCC computation
- Checkpointing (best model and last model) and early stopping
- Comprehensive metric logging (Total loss, Span loss, Valence/Arousal CCC, Pearson, MAE, RMSE)
"""

import json
import logging
import math
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..losses.va_loss import JointVALoss
from ..losses.joint_loss import JointLoss
from ..losses.span_loss import BiaffineSpanLoss
from ..evaluation.metrics import evaluate_predictions, compute_span_metrics
from ..utils.checkpoint import CheckpointManager
from ..utils.device import get_device

logger = logging.getLogger(__name__)


def build_optimizer(
    model: nn.Module,
    learning_rate: float = 1.0e-4,
    backbone_learning_rate: float = 2.0e-5,
    weight_decay: float = 0.01,
) -> torch.optim.Optimizer:
    """Build AdamW optimizer with 2 distinct learning rate groups: backbone vs new modules."""
    backbone_params = []
    new_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            new_params.append(param)

    optimizer_grouped_parameters = [
        {"params": backbone_params, "lr": backbone_learning_rate, "weight_decay": weight_decay},
        {"params": new_params, "lr": learning_rate, "weight_decay": weight_decay},
    ]
    return torch.optim.AdamW(optimizer_grouped_parameters)


def build_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Build linear warmup and linear decay learning rate scheduler."""
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0,
            float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)),
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class NSSGDimNetTrainer:
    """Trainer coordinating optimization, evaluation, and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Optional[Union[JointVALoss, JointLoss]] = None,
        span_loss_fn: Optional[BiaffineSpanLoss] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        device: Optional[torch.device] = None,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
        mixed_precision: bool = False,
        early_stopping_patience: int = 7,
        lambda_span: float = 0.5,
        aspect_pos_weight: float = 50.0,
        opinion_pos_weight: float = 50.0,
    ):
        self.device = device or get_device("auto")
        self.model = model.to(self.device)
        self.loss_fn = loss_fn or JointVALoss()
        self.span_loss_fn = span_loss_fn or BiaffineSpanLoss(
            aspect_pos_weight=aspect_pos_weight,
            opinion_pos_weight=opinion_pos_weight,
        )
        self.lambda_span = lambda_span
        self.aspect_pos_weight = aspect_pos_weight
        self.opinion_pos_weight = opinion_pos_weight

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_manager = checkpoint_manager
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self.mixed_precision = mixed_precision and self.device.type == "cuda"
        self.early_stopping_patience = early_stopping_patience

        self.scaler = torch.amp.GradScaler("cuda", enabled=self.mixed_precision)
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_loss_va": [],
            "train_loss_span": [],
            "train_loss_asp_span": [],
            "train_loss_op_span": [],
            "train_mean_pos_asp_logit": [],
            "train_mean_neg_asp_logit": [],
            "train_mean_pos_op_logit": [],
            "train_mean_neg_op_logit": [],
            "val_loss": [],
            "val_pearson_v": [],
            "val_pearson_a": [],
            "val_pearson_mean": [],
            "val_mae_v": [],
            "val_mae_a": [],
            "val_ccc_v": [],
            "val_ccc_a": [],
            "val_ccc_mean": [],
            "val_aspect_precision": [],
            "val_aspect_recall": [],
            "val_aspect_f1": [],
            "val_opinion_precision": [],
            "val_opinion_recall": [],
            "val_opinion_f1": [],
        }

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform a single forward and backward optimization step with wired span loss."""
        self.model.train()
        if self.optimizer is not None:
            self.optimizer.zero_grad()

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        lang_ids = batch.get("lang_ids", None)
        if lang_ids is not None:
            lang_ids = lang_ids.to(self.device)

        aspect_spans = batch.get("aspect_spans", None)
        if aspect_spans is not None:
            aspect_spans = aspect_spans.to(self.device)

        opinion_spans = batch.get("opinion_spans", None)
        if opinion_spans is not None:
            opinion_spans = opinion_spans.to(self.device)

        target_v = batch["valence"].to(self.device)
        target_a = batch["arousal"].to(self.device)

        forward_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "aspect_spans": aspect_spans,
            "opinion_spans": opinion_spans,
        }
        if lang_ids is not None:
            forward_kwargs["lang_ids"] = lang_ids

        outputs = self.model(**forward_kwargs)
        va_loss_dict = self.loss_fn(
            pred_valence=outputs["valence"],
            pred_arousal=outputs["arousal"],
            target_valence=target_v,
            target_arousal=target_a,
        )
        l_va = va_loss_dict["total_loss"]

        # Compute span loss if model outputs span scores and targets exist in batch
        l_span = torch.tensor(0.0, device=self.device)
        l_asp_span = torch.tensor(0.0, device=self.device)
        l_op_span = torch.tensor(0.0, device=self.device)

        if outputs.get("aspect_scores") is not None and batch.get("aspect_span_targets") is not None:
            valid_mask = outputs.get("valid_span_mask")
            if valid_mask is None:
                valid_mask = batch.get("valid_span_mask", torch.ones_like(batch["aspect_span_targets"], dtype=torch.bool))

            span_loss_dict = self.span_loss_fn(
                aspect_scores=outputs["aspect_scores"],
                opinion_scores=outputs["opinion_scores"],
                valid_span_mask=valid_mask.to(self.device),
                gold_aspect_spans=batch["aspect_span_targets"].to(self.device),
                gold_opinion_spans=batch["opinion_span_targets"].to(self.device),
                aspect_supervision_mask=batch.get("aspect_supervision_mask").to(self.device) if batch.get("aspect_supervision_mask") is not None else None,
                opinion_supervision_mask=batch.get("opinion_supervision_mask").to(self.device) if batch.get("opinion_supervision_mask") is not None else None,
            )
            l_asp_span = span_loss_dict["aspect_span_loss"]
            l_op_span = span_loss_dict["opinion_span_loss"]
            l_span = l_asp_span + l_op_span

        total_loss = l_va + self.lambda_span * l_span
        total_loss.backward()

        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        if self.optimizer is not None:
            self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        return {
            "total_loss": total_loss.item(),
            "loss_va": l_va.item(),
            "loss_span": l_span.item(),
            "loss_asp_span": l_asp_span.item(),
            "loss_op_span": l_op_span.item(),
            **{k: v.item() for k, v in va_loss_dict.items() if k != "total_loss"}
        }

    def train_epoch(self, dataloader: DataLoader, epoch: int = 0) -> Dict[str, float]:
        """Train model for one epoch with comprehensive logging of VA, span, and logit metrics."""
        self.model.train()
        total_loss = 0.0
        total_va_loss = 0.0
        total_span_loss = 0.0
        total_asp_span_loss = 0.0
        total_op_span_loss = 0.0
        steps = 0

        pos_asp_logits_accum = []
        neg_asp_logits_accum = []
        pos_op_logits_accum = []
        neg_op_logits_accum = []

        accum_preds_v = []
        accum_preds_a = []
        accum_targets_v = []
        accum_targets_a = []

        if self.optimizer is not None:
            self.optimizer.zero_grad()

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            lang_ids = batch.get("lang_ids", None)
            if lang_ids is not None:
                lang_ids = lang_ids.to(self.device)

            aspect_spans = batch.get("aspect_spans", None)
            if aspect_spans is not None:
                aspect_spans = aspect_spans.to(self.device)

            opinion_spans = batch.get("opinion_spans", None)
            if opinion_spans is not None:
                opinion_spans = opinion_spans.to(self.device)

            target_v = batch["valence"].to(self.device)
            target_a = batch["arousal"].to(self.device)

            forward_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "aspect_spans": aspect_spans,
                "opinion_spans": opinion_spans,
            }
            if lang_ids is not None:
                forward_kwargs["lang_ids"] = lang_ids

            with torch.amp.autocast("cuda", enabled=self.mixed_precision):
                outputs = self.model(**forward_kwargs)
                pred_v = outputs["valence"]
                pred_a = outputs["arousal"]

                va_loss_dict = self.loss_fn(
                    pred_valence=pred_v,
                    pred_arousal=pred_a,
                    target_valence=target_v,
                    target_arousal=target_a,
                )
                l_va = va_loss_dict["total_loss"]

                l_span = torch.tensor(0.0, device=self.device)
                l_asp_span = torch.tensor(0.0, device=self.device)
                l_op_span = torch.tensor(0.0, device=self.device)

                if outputs.get("aspect_scores") is not None and batch.get("aspect_span_targets") is not None:
                    valid_mask = outputs.get("valid_span_mask")
                    if valid_mask is None:
                        valid_mask = batch.get("valid_span_mask", torch.ones_like(batch["aspect_span_targets"], dtype=torch.bool))

                    span_loss_dict = self.span_loss_fn(
                        aspect_scores=outputs["aspect_scores"],
                        opinion_scores=outputs["opinion_scores"],
                        valid_span_mask=valid_mask.to(self.device),
                        gold_aspect_spans=batch["aspect_span_targets"].to(self.device),
                        gold_opinion_spans=batch["opinion_span_targets"].to(self.device),
                        aspect_supervision_mask=batch.get("aspect_supervision_mask").to(self.device) if batch.get("aspect_supervision_mask") is not None else None,
                        opinion_supervision_mask=batch.get("opinion_supervision_mask").to(self.device) if batch.get("opinion_supervision_mask") is not None else None,
                    )
                    l_asp_span = span_loss_dict["aspect_span_loss"]
                    l_op_span = span_loss_dict["opinion_span_loss"]
                    l_span = l_asp_span + l_op_span

                    # Track positive vs negative logits
                    with torch.no_grad():
                        asp_s = outputs["aspect_scores"]
                        op_s = outputs["opinion_scores"]
                        asp_t = batch["aspect_span_targets"].to(self.device)
                        op_t = batch["opinion_span_targets"].to(self.device)
                        v_m = valid_mask.to(self.device)

                        pos_asp_m = v_m & (asp_t == 1.0)
                        neg_asp_m = v_m & (asp_t == 0.0)
                        pos_op_m = v_m & (op_t == 1.0)
                        neg_op_m = v_m & (op_t == 0.0)

                        if pos_asp_m.any():
                            pos_asp_logits_accum.append(asp_s[pos_asp_m].mean().item())
                        if neg_asp_m.any():
                            neg_asp_logits_accum.append(asp_s[neg_asp_m].mean().item())
                        if pos_op_m.any():
                            pos_op_logits_accum.append(op_s[pos_op_m].mean().item())
                        if neg_op_m.any():
                            neg_op_logits_accum.append(op_s[neg_op_m].mean().item())

                total_batch_loss = l_va + self.lambda_span * l_span
                loss = total_batch_loss / self.gradient_accumulation_steps

            if self.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % self.gradient_accumulation_steps == 0 or (step + 1) == len(dataloader):
                if self.mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                    if self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    if self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    if self.optimizer is not None:
                        self.optimizer.step()

                if self.optimizer is not None:
                    self.optimizer.zero_grad()
                if self.scheduler is not None:
                    self.scheduler.step()

            total_loss += total_batch_loss.item()
            total_va_loss += l_va.item()
            total_span_loss += l_span.item()
            total_asp_span_loss += l_asp_span.item()
            total_op_span_loss += l_op_span.item()
            steps += 1

            # Accumulate predictions and targets for epoch-level statistics
            accum_preds_v.extend(pred_v.detach().cpu().view(-1).tolist())
            accum_preds_a.extend(pred_a.detach().cpu().view(-1).tolist())
            accum_targets_v.extend(target_v.detach().cpu().view(-1).tolist())
            accum_targets_a.extend(target_a.detach().cpu().view(-1).tolist())

        avg_loss = total_loss / max(1, steps)
        avg_va_loss = total_va_loss / max(1, steps)
        avg_span_loss = total_span_loss / max(1, steps)
        avg_asp_span_loss = total_asp_span_loss / max(1, steps)
        avg_op_span_loss = total_op_span_loss / max(1, steps)

        train_metrics = evaluate_predictions(accum_targets_v, accum_preds_v, accum_targets_a, accum_preds_a)
        train_metrics["loss"] = avg_loss
        train_metrics["loss_va"] = avg_va_loss
        train_metrics["loss_span"] = avg_span_loss
        train_metrics["loss_asp_span"] = avg_asp_span_loss
        train_metrics["loss_op_span"] = avg_op_span_loss
        train_metrics["mean_pos_asp_logit"] = float(np.mean(pos_asp_logits_accum)) if pos_asp_logits_accum else 0.0
        train_metrics["mean_neg_asp_logit"] = float(np.mean(neg_asp_logits_accum)) if neg_asp_logits_accum else 0.0
        train_metrics["mean_pos_op_logit"] = float(np.mean(pos_op_logits_accum)) if pos_op_logits_accum else 0.0
        train_metrics["mean_neg_op_logit"] = float(np.mean(neg_op_logits_accum)) if neg_op_logits_accum else 0.0
        return train_metrics

    def evaluate(self, dataloader: DataLoader, mode: str = "end_to_end") -> Dict[str, float]:
        """Evaluate model on validation or test dataset with epoch-level CCC and span metrics."""
        self.model.eval()
        total_loss = 0.0
        steps = 0

        preds_v, preds_a = [], []
        targets_v, targets_a = [], []
        gold_aspects_accum = []
        pred_aspects_accum = []
        gold_opinions_accum = []
        pred_opinions_accum = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                lang_ids = batch.get("lang_ids", None)
                if lang_ids is not None:
                    lang_ids = lang_ids.to(self.device)

                # In oracle mode, pass gold spans; in end-to-end mode, leave None so model predicts
                if mode == "oracle":
                    aspect_spans = batch.get("aspect_spans", None).to(self.device) if batch.get("aspect_spans") is not None else None
                    opinion_spans = batch.get("opinion_spans", None).to(self.device) if batch.get("opinion_spans") is not None else None
                else:
                    aspect_spans = None
                    opinion_spans = None

                target_v = batch["valence"].to(self.device)
                target_a = batch["arousal"].to(self.device)

                forward_kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "aspect_spans": aspect_spans,
                    "opinion_spans": opinion_spans,
                }
                if lang_ids is not None:
                    forward_kwargs["lang_ids"] = lang_ids

                outputs = self.model(**forward_kwargs)
                pred_v = outputs["valence"]
                pred_a = outputs["arousal"]

                loss_dict = self.loss_fn(
                    pred_valence=pred_v,
                    pred_arousal=pred_a,
                    target_valence=target_v,
                    target_arousal=target_a,
                )
                total_loss += loss_dict["total_loss"].item()
                steps += 1

                preds_v.extend(pred_v.cpu().view(-1).tolist())
                preds_a.extend(pred_a.cpu().view(-1).tolist())
                targets_v.extend(target_v.cpu().view(-1).tolist())
                targets_a.extend(target_a.cpu().view(-1).tolist())

                dec_a = outputs.get("decoded_aspect_spans", [])
                dec_o = outputs.get("decoded_opinion_spans", [])

                batch_asp_spans = batch.get("aspect_spans")
                batch_op_spans = batch.get("opinion_spans")

                for b in range(len(pred_v)):
                    g_a = [(int(batch_asp_spans[b, 0].item()), int(batch_asp_spans[b, 1].item()))] if batch_asp_spans is not None else []
                    g_o = [(int(batch_op_spans[b, 0].item()), int(batch_op_spans[b, 1].item()))] if batch_op_spans is not None else []
                    gold_aspects_accum.append(g_a)
                    gold_opinions_accum.append(g_o)

                    p_a = [(d["start"], d["end"]) for d in (dec_a[b] if b < len(dec_a) else [])]
                    p_o = [(d["start"], d["end"]) for d in (dec_o[b] if b < len(dec_o) else [])]
                    pred_aspects_accum.append(p_a)
                    pred_opinions_accum.append(p_o)

        metrics = evaluate_predictions(targets_v, preds_v, targets_a, preds_a)
        asp_m = compute_span_metrics(gold_aspects_accum, pred_aspects_accum)
        op_m = compute_span_metrics(gold_opinions_accum, pred_opinions_accum)

        metrics["loss"] = total_loss / max(1, steps)
        metrics["aspect_precision"] = asp_m["precision"]
        metrics["aspect_recall"] = asp_m["recall"]
        metrics["aspect_f1"] = asp_m["f1"]
        metrics["opinion_precision"] = op_m["precision"]
        metrics["opinion_recall"] = op_m["recall"]
        metrics["opinion_f1"] = op_m["f1"]
        return metrics

    def fit(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        epochs: int = 15,
        early_stopping_patience: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute complete multi-epoch training loop with early stopping."""
        patience = early_stopping_patience or self.early_stopping_patience
        best_metric = -float("inf")
        best_epoch = 0
        patience_counter = 0

        logger.info("Starting training for %d epochs...", epochs)

        for epoch in range(epochs):
            train_res = self.train_epoch(train_dataloader, epoch=epoch)
            val_res = self.evaluate(val_dataloader)

            # Combined validation metric: (Pearson_V + Pearson_A) / 2
            val_combined_pearson = val_res.get("pearson_mean", (val_res.get("pearson_v", 0.0) + val_res.get("pearson_a", 0.0)) / 2.0)
            val_res["val_combined_pearson"] = val_combined_pearson

            # Update history
            self.history["train_loss"].append(train_res["loss"])
            self.history["train_loss_va"].append(train_res.get("loss_va", train_res["loss"]))
            self.history["train_loss_span"].append(train_res.get("loss_span", 0.0))
            self.history["train_loss_asp_span"].append(train_res.get("loss_asp_span", 0.0))
            self.history["train_loss_op_span"].append(train_res.get("loss_op_span", 0.0))
            self.history["train_mean_pos_asp_logit"].append(train_res.get("mean_pos_asp_logit", 0.0))
            self.history["train_mean_neg_asp_logit"].append(train_res.get("mean_neg_asp_logit", 0.0))
            self.history["train_mean_pos_op_logit"].append(train_res.get("mean_pos_op_logit", 0.0))
            self.history["train_mean_neg_op_logit"].append(train_res.get("mean_neg_op_logit", 0.0))

            self.history["val_loss"].append(val_res["loss"])
            self.history["val_pearson_v"].append(val_res["pearson_v"])
            self.history["val_pearson_a"].append(val_res["pearson_a"])
            self.history["val_pearson_mean"].append(val_combined_pearson)
            self.history["val_mae_v"].append(val_res["mae_v"])
            self.history["val_mae_a"].append(val_res["mae_a"])
            self.history["val_ccc_v"].append(val_res["ccc_v"])
            self.history["val_ccc_a"].append(val_res["ccc_a"])
            self.history["val_ccc_mean"].append(val_res["ccc_mean"])

            self.history["val_aspect_precision"].append(val_res["aspect_precision"])
            self.history["val_aspect_recall"].append(val_res["aspect_recall"])
            self.history["val_aspect_f1"].append(val_res["aspect_f1"])
            self.history["val_opinion_precision"].append(val_res["opinion_precision"])
            self.history["val_opinion_recall"].append(val_res["opinion_recall"])
            self.history["val_opinion_f1"].append(val_res["opinion_f1"])

            logger.info(
                "Epoch %d/%d - Loss (Tot/VA/Span): %.4f/%.4f/%.4f | Val Pearson: %.3f/%.3f (Mean: %.3f) | Val F1 (Asp/Op): %.4f/%.4f",
                epoch + 1, epochs, train_res["loss"], train_res.get("loss_va", 0.0), train_res.get("loss_span", 0.0),
                val_res["pearson_v"], val_res["pearson_a"], val_combined_pearson,
                val_res["aspect_f1"], val_res["opinion_f1"]
            )

            # Checkpoint saving
            if self.checkpoint_manager is not None:
                self.checkpoint_manager.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch + 1,
                    metrics=val_res,
                )

            # Early stopping check
            if val_combined_pearson > best_metric:
                best_metric = val_combined_pearson
                best_epoch = epoch + 1
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping triggered at epoch %d (Best Epoch %d)", epoch + 1, best_epoch)
                    break

        return {
            "best_epoch": best_epoch,
            "best_metric": best_metric,
            "history": self.history,
        }
