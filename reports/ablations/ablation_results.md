# NSSG-DimNet Ablation Study Results

| Ablation Study | Description | Valence $r$ | Valence MAE | Arousal $r$ | Arousal MAE | Combined $r$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `wo_switch_embedding` | Removes language ID and signed switch-distance embeddings; feeds pure backbone output to attention. | -0.0150 | 0.2032 | -0.1030 | 0.1354 | -0.0590 |
| `wo_sp_gsa` | Replaces SP-GSA with standard transformer self-attention without switch-point gating. | -0.0150 | 0.2032 | -0.1030 | 0.1354 | -0.0590 |
| `wo_hnsg_rgat` | Removes Branch 2 (H-NSG + RGAT); regresses directly from Branch 1 span representation. | -0.0068 | 0.2056 | 0.1434 | 0.1197 | 0.0683 |
| `wo_nrc_vad` | Excludes symbolic NRC-VAD affective prior nodes and edges from H-NSG graph. | -0.0150 | 0.2032 | -0.1030 | 0.1354 | -0.0590 |
| `wo_cross_attention` | Replaces mutual cross-attention with simple naive concatenation of span and graph vectors. | -0.0150 | 0.2032 | -0.1030 | 0.1354 | -0.0590 |
| `wo_ccc_loss` | Ablates the CCC loss component, training solely on Smooth L1 regression loss. | -0.0067 | 0.1894 | -0.1800 | 0.1225 | -0.0934 |
