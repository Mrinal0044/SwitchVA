import torch
import torch.nn as nn
from transformers import AutoModel

class SwitchVANet(nn.Module):
    def __init__(self, encoder_name='xlm-roberta-base', use_spe=True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.use_spe = use_spe
        
        # Code-Switch Locator (CSL) token classification head (3 classes: 0=Hindi, 1=English, 2=Other)
        self.csl_head = nn.Linear(self.hidden_size, 3)
        
        # Switch-Proximity Encoding (SPE) MLP
        if self.use_spe:
            self.spe_mlp = nn.Sequential(
                nn.Linear(4, 16),
                nn.ReLU(),
                nn.Linear(16, 32),
                nn.ReLU()
            )
            arousal_in_size = self.hidden_size + 32
        else:
            arousal_in_size = self.hidden_size
            
        # Regression heads for Valence and Arousal (Asymmetric Fusion)
        self.valence_head = nn.Linear(self.hidden_size, 1)
        self.arousal_head = nn.Linear(arousal_in_size, 1)
        
    def forward(self, input_ids, attention_mask, aspect_token_mask, spe_features=None):
        # 1. Forward pass through multilingual encoder backbone
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state # Shape: (batch_size, sequence_length, hidden_size)
        
        # 2. Token-level language predictions (CSL)
        csl_logits = self.csl_head(hidden_states) # Shape: (batch_size, sequence_length, 3)
        
        # 3. Aspect Span Pooling (average pooling over mask)
        # aspect_token_mask shape: (batch_size, sequence_length)
        mask_expanded = aspect_token_mask.unsqueeze(-1) # (batch_size, sequence_length, 1)
        sum_mask = aspect_token_mask.sum(dim=-1, keepdim=True) # (batch_size, 1)
        sum_mask = torch.clamp(sum_mask, min=1e-9) # Prevent division by zero
        
        pooled_aspect = (hidden_states * mask_expanded).sum(dim=1) / sum_mask # Shape: (batch_size, hidden_size)
        
        # 4. Predict Valence (uses only the encoder aspect representation)
        valence = self.valence_head(pooled_aspect).squeeze(-1) # Shape: (batch_size,)
        
        # 5. Predict Arousal (asymmetric fusion: Valence head does not see SPE, Arousal head does)
        if self.use_spe and spe_features is not None:
            spe_embedded = self.spe_mlp(spe_features) # Shape: (batch_size, 32)
            arousal_input = torch.cat([pooled_aspect, spe_embedded], dim=-1) # (batch_size, hidden_size + 32)
        else:
            arousal_input = pooled_aspect # (batch_size, hidden_size)
            
        arousal = self.arousal_head(arousal_input).squeeze(-1) # Shape: (batch_size,)
        
        return {
            'csl_logits': csl_logits,
            'valence': valence,
            'arousal': arousal
        }

if __name__ == "__main__":
    # Check compilation and simple forward pass
    model = SwitchVANet(encoder_name='xlm-roberta-base', use_spe=True)
    print("Model initialized successfully!")
    
    # Fake batch
    batch_size = 2
    seq_len = 128
    dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    dummy_mask = torch.ones((batch_size, seq_len))
    dummy_aspect_mask = torch.zeros((batch_size, seq_len))
    dummy_aspect_mask[:, 10:15] = 1.0 # aspect spans from index 10 to 14
    dummy_spe = torch.randn((batch_size, 4))
    
    out = model(dummy_input_ids, dummy_mask, dummy_aspect_mask, dummy_spe)
    print("Forward pass successful!")
    print("CSL logits shape:", out['csl_logits'].shape)
    print("Valence shape:", out['valence'].shape)
    print("Arousal shape:", out['arousal'].shape)
