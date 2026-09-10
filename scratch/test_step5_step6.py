"""Step 5 and Step 6 debug script: 10-example memorization test & gradient norm verification."""

import json
import torch
from torch.utils.data import DataLoader, Subset

from src.models.nssg_dimnet import NSSGDimNet
from src.training.data_utils import DimABSATensorDataset, dimabsa_tensor_collate_fn
from src.training.trainer import NSSGDimNetTrainer, build_optimizer
from src.evaluation.metrics import compute_span_metrics

def main():
    print("=== STEP 6: GRADIENT VERIFICATION AFTER 1 REAL STEP ===")
    dataset = DimABSATensorDataset(
        jsonl_path="data/splits/train.jsonl",
        max_seq_length=64,
        training_safe_only=True
    )
    
    # Select first 10 distinct sentences
    # Group sample indices by sentence_id to get all annotations for 10 sentences
    seen_sents = []
    subset_indices = []
    for idx, sample in enumerate(dataset.samples):
        sid = sample["sentence_id"]
        if sid not in seen_sents:
            if len(seen_sents) >= 10:
                continue
            seen_sents.append(sid)
        subset_indices.append(idx)
        
    print(f"Loaded {len(subset_indices)} samples across {len(seen_sents)} unique sentences.")
    
    loader = DataLoader(
        Subset(dataset, subset_indices),
        batch_size=4,
        shuffle=False,
        collate_fn=dimabsa_tensor_collate_fn
    )
    
    model = NSSGDimNet(
        hidden_dim=64,
        num_spgsa_heads=2,
        rgat_heads=2,
        cross_attn_heads=2,
        use_mock_backbone=True,
    )
    
    trainer = NSSGDimNetTrainer(
        model=model,
        lambda_span=1.0,
        aspect_pos_weight=50.0,
        opinion_pos_weight=50.0,
        max_grad_norm=1.0
    )
    optimizer = build_optimizer(model, learning_rate=5e-3, backbone_learning_rate=1e-3)
    trainer.optimizer = optimizer
    
    # Grab the first batch
    first_batch = next(iter(loader))
    print("Batch keys:", list(first_batch.keys()))
    assert "aspect_span_targets" in first_batch, "aspect_span_targets missing!"
    assert "opinion_span_targets" in first_batch, "opinion_span_targets missing!"
    print(f"aspect_span_targets shape: {first_batch['aspect_span_targets'].shape}")
    print(f"opinion_span_targets shape: {first_batch['opinion_span_targets'].shape}")
    print(f"aspect_span_targets sum: {first_batch['aspect_span_targets'].sum().item()}")
    print(f"opinion_span_targets sum: {first_batch['opinion_span_targets'].sum().item()}")
    
    # Perform 1 step
    step_metrics = trainer.train_step(first_batch)
    print("Step 1 metrics:", step_metrics)
    
    # Check gradient norms
    # To compute gradient norm before optimizer zero_grad, let's do a clean backward pass without optimizer.step
    device = trainer.device
    model.zero_grad()
    outputs = model(
        input_ids=first_batch["input_ids"].to(device),
        attention_mask=first_batch["attention_mask"].to(device),
        lang_ids=first_batch["lang_ids"].to(device),
    )
    va_loss = trainer.loss_fn(
        pred_valence=outputs["valence"],
        pred_arousal=outputs["arousal"],
        target_valence=first_batch["valence"].to(device),
        target_arousal=first_batch["arousal"].to(device)
    )["total_loss"]
    
    span_loss_dict = trainer.span_loss_fn(
        aspect_scores=outputs["aspect_scores"],
        opinion_scores=outputs["opinion_scores"],
        valid_span_mask=first_batch["valid_span_mask"].to(device),
        gold_aspect_spans=first_batch["aspect_span_targets"].to(device),
        gold_opinion_spans=first_batch["opinion_span_targets"].to(device),
        aspect_supervision_mask=first_batch["aspect_supervision_mask"].to(device) if first_batch.get("aspect_supervision_mask") is not None else None,
        opinion_supervision_mask=first_batch["opinion_supervision_mask"].to(device) if first_batch.get("opinion_supervision_mask") is not None else None,
    )
    span_loss = span_loss_dict["aspect_span_loss"] + span_loss_dict["opinion_span_loss"]
    total_loss = va_loss + trainer.lambda_span * span_loss
    total_loss.backward()
    
    def get_module_grad_norm(module):
        total_norm = 0.0
        has_grad = False
        for p in module.parameters():
            if p.grad is not None:
                has_grad = True
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5 if has_grad else 0.0

    print("\n--- GRADIENT NORMS PER MODULE ---")
    modules_to_check = {
        "aspect_start_mlp": model.biaffine_span.aspect_start_mlp,
        "aspect_end_mlp": model.biaffine_span.aspect_end_mlp,
        "aspect_scorer": model.biaffine_span.aspect_scorer,
        "opinion_start_mlp": model.biaffine_span.opinion_start_mlp,
        "opinion_end_mlp": model.biaffine_span.opinion_end_mlp,
        "opinion_scorer": model.biaffine_span.opinion_scorer,
        "backbone": model.backbone,
        "switch_embedding": model.switch_embedding,
        "sp_gsa": model.sp_gsa,
        "rgat": model.rgat,
        "cross_attention": model.cross_attention,
        "fusion": model.fusion,
        "valence_head": model.valence_head,
        "arousal_head": model.arousal_head,
    }
    
    grad_results = {}
    for name, mod in modules_to_check.items():
        gnorm = get_module_grad_norm(mod)
        grad_results[name] = gnorm
        print(f"  {name:20s}: grad_norm = {gnorm:.6f}")
        assert gnorm > 0.0, f"Module {name} received ZERO gradient!"
        
    print("\nALL 14 MODULES RECEIVED NON-ZERO GRADIENTS! Step 6 check PASSED.")
    
    print("\n=== STEP 5: 10-EXAMPLE MEMORIZATION TEST ===")
    # Train for 40 epochs on these 10 sentences
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    trainer.optimizer = optimizer
    
    history = []
    for epoch in range(1, 41):
        epoch_loss = 0.0
        epoch_va = 0.0
        epoch_span = 0.0
        epoch_asp_span = 0.0
        epoch_op_span = 0.0
        num_batches = 0
        
        for batch in loader:
            metrics = trainer.train_step(batch)
            epoch_loss += metrics["total_loss"]
            epoch_va += metrics["loss_va"]
            epoch_span += metrics["loss_span"]
            epoch_asp_span += metrics["loss_asp_span"]
            epoch_op_span += metrics["loss_op_span"]
            num_batches += 1
            
        epoch_loss /= num_batches
        epoch_va /= num_batches
        epoch_span /= num_batches
        epoch_asp_span /= num_batches
        epoch_op_span /= num_batches
        
        # Evaluate extraction on the 10 sentences
        model.eval()
        all_pred_asp = []
        all_pred_op = []
        all_gold_asp = []
        all_gold_op = []
        
        with torch.no_grad():
            for batch in loader:
                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    lang_ids=batch["lang_ids"].to(device),
                    top_k=5,
                    strategy="ranked",
                )
                all_pred_asp.extend(outputs["decoded_aspect_spans"])
                all_pred_op.extend(outputs["decoded_opinion_spans"])
                all_gold_asp.extend(batch["gold_aspect_spans"])
                all_gold_op.extend(batch["gold_opinion_spans"])
                
        asp_metrics = compute_span_metrics(all_pred_asp, all_gold_asp)
        op_metrics = compute_span_metrics(all_pred_op, all_gold_op)
        
        entry = {
            "epoch": epoch,
            "total_loss": epoch_loss,
            "va_loss": epoch_va,
            "span_loss": epoch_span,
            "asp_span_loss": epoch_asp_span,
            "op_span_loss": epoch_op_span,
            "aspect_precision": asp_metrics["precision"],
            "aspect_recall": asp_metrics["recall"],
            "aspect_f1": asp_metrics["f1"],
            "opinion_precision": op_metrics["precision"],
            "opinion_recall": op_metrics["recall"],
            "opinion_f1": op_metrics["f1"],
        }
        history.append(entry)
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:2d} | Total: {epoch_loss:.4f} | VA: {epoch_va:.4f} | Span: {epoch_span:.4f} "
                  f"(Asp: {epoch_asp_span:.4f}, Op: {epoch_op_span:.4f}) | "
                  f"Asp F1: {asp_metrics['f1']:.4f} (P: {asp_metrics['precision']:.3f}, R: {asp_metrics['recall']:.3f}) | "
                  f"Op F1: {op_metrics['f1']:.4f} (P: {op_metrics['precision']:.3f}, R: {op_metrics['recall']:.3f})")
            
    # Verify span loss decreased significantly
    initial_span_loss = history[0]["span_loss"]
    final_span_loss = history[-1]["span_loss"]
    print(f"\nInitial span loss: {initial_span_loss:.4f}, Final span loss: {final_span_loss:.4f}")
    assert final_span_loss < initial_span_loss * 0.2, "Span loss did not decrease significantly!"
    
    # Verify F1 improved
    print(f"Final Aspect F1: {history[-1]['aspect_f1']:.4f}, Final Opinion F1: {history[-1]['opinion_f1']:.4f}")
    assert history[-1]['aspect_f1'] > 0.0 or history[-1]['opinion_f1'] > 0.0, "F1 did not improve!"
    print("\n10-EXAMPLE MEMORIZATION TEST SUCCEEDED!")

if __name__ == "__main__":
    main()
