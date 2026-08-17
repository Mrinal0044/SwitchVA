import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

def run_mixed_effects():
    baseline_path = '/Users/kmrinal/SwitchVA/baseline_predictions.csv'
    if not os.path.exists(baseline_path):
        print(f"Error: predictions file not found at {baseline_path}. Please run train.py first.")
        return
        
    df = pd.read_csv(baseline_path)
    
    # 1. Compute baseline model prediction errors
    df['arousal_error'] = np.abs(df['predicted_arousal'] - df['computed_arousal'])
    df['valence_error'] = np.abs(df['predicted_valence'] - df['computed_valence'])
    
    print("\n" + "="*50)
    print("CRITICAL EXPERIMENT: MIXED-EFFECTS REGRESSION ANALYSIS")
    print("="*50)
    print(f"Loaded test predictions: {len(df)} samples.")
    print(f"Average Arousal Error: {df['arousal_error'].mean():.4f}")
    print(f"Average Valence Error: {df['valence_error'].mean():.4f}")
    
    # 2. Regression 1: On the subset of sentences with switches (where distance < 128)
    df_switched = df[df['spe_distance'] < 128].copy()
    print(f"\n--- Substudy 1: Switched sentences only ({len(df_switched)} samples) ---")
    print("Formula: arousal_error ~ spe_distance + spe_density")
    print("Random Intercept: (1 | domain)")
    
    if len(df_switched) > 5:
        # Fit Mixed Linear Model
        model_sw = smf.mixedlm("arousal_error ~ spe_distance + spe_density", data=df_switched, groups=df_switched["domain"])
        result_sw = model_sw.fit()
        print(result_sw.summary())
    else:
        print("Not enough switched samples in the test split to run regression.")
        
    # 3. Regression 2: On the full dataset, converting distance to proximity = 1 / (1 + distance)
    # This handles sentences without switches gracefully (distance = 999.0 becomes proximity = 0.001)
    df['spe_proximity'] = 1.0 / (1.0 + df['spe_distance'])
    print(f"\n--- Substudy 2: Full dataset ({len(df)} samples) using Switch Proximity ---")
    print("Formula: arousal_error ~ spe_proximity + spe_density")
    print("Random Intercept: (1 | domain)")
    
    model_full = smf.mixedlm("arousal_error ~ spe_proximity + spe_density", data=df, groups=df["domain"])
    result_full = model_full.fit()
    print(result_full.summary())
    
    # Check if distance is significant for arousal vs valence
    print("\n--- Control Experiment: Valence prediction error ~ switch features ---")
    model_v = smf.mixedlm("valence_error ~ spe_proximity + spe_density", data=df, groups=df["domain"])
    result_v = model_v.fit()
    print(result_v.summary())
    
    # Print key paper-ready takeaway
    p_val_arousal = result_full.pvalues['spe_proximity']
    coef_arousal = result_full.params['spe_proximity']
    p_val_val = result_v.pvalues['spe_proximity']
    
    print("\n" + "="*50)
    print("SCIENTIFIC TAKEAWAY:")
    print(f"Arousal Error Proximity Coeff: {coef_arousal:.4f} (p = {p_val_arousal:.4f})")
    print(f"Valence Error Proximity Coeff: {result_v.params['spe_proximity']:.4f} (p = {p_val_val:.4f})")
    
    if p_val_arousal < 0.05:
        print("\nSUCCESS: Code-switch proximity is a statistically significant predictor of baseline arousal error (p < 0.05)!")
        print("This supports the paper's core hypothesis that switching patterns carry arousal-relevant signals that the baseline model ignores.")
    else:
        print("\nNote: Code-switch proximity is not statistically significant (p >= 0.05) under this validation run.")
    print("="*50 + "\n")

if __name__ == '__main__':
    run_mixed_effects()
