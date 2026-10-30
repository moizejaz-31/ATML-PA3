"""
Data loading and parquet joining for Task 3 rollout tables.
Joins generations.parquet, probability_metrics.parquet, and attention_metrics.parquet
on (dataset, sample_id, rollout_id).
"""
import os
import pandas as pd

def load_and_join_rollouts(folder_path: str, is_vlm: bool = False) -> pd.DataFrame:
    """
    Load parquet tables from model folder and join them.
    """
    gen_path = os.path.join(folder_path, "generations.parquet")
    prob_path = os.path.join(folder_path, "probability_metrics.parquet")
    
    df_gen = pd.read_parquet(gen_path)
    df_prob = pd.read_parquet(prob_path)
    join_keys = ["dataset", "sample_id", "rollout_id"]
    
    df = pd.merge(df_gen, df_prob, on=join_keys)
    
    if is_vlm:
        att_path = os.path.join(folder_path, "attention_metrics.parquet")
        df_att = pd.read_parquet(att_path)
        df = pd.merge(df, df_att, on=join_keys)
        
    return df
