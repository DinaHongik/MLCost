#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
import faiss
import json
import logging
from typing import Dict, List
import re

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class HardwareKnowledgeBase:
    
    def __init__(self):
        self.gpu_scores = {
            'b200': 3.00, 'blackwell': 3.00,
            'h200': 2.50, 'h100': 2.20,
            'a100': 1.00, 'l40s': 0.85, 'a30': 0.55,
            'v100': 0.45, 'rtx-4090': 0.35, '4090': 0.35,
            'l4': 0.25, 'tpu-v5e': 2.00, 'tpu-v5': 2.00,
            'tpu-v4': 1.20, 'gaudi2': 0.90,
            'bow-ipu': 0.50, 'ipu': 0.50,
        }
        
        self.benchmark_profiles = {
            'bert': {'compute': 0.70, 'memory': 0.60, 'scaling': 0.75},
            'resnet': {'compute': 0.90, 'memory': 0.30, 'scaling': 0.85},
            'dlrm': {'compute': 0.40, 'memory': 0.90, 'scaling': 0.60},
            'gpt3': {'compute': 0.80, 'memory': 0.70, 'scaling': 0.65},
            'maskrcnn': {'compute': 0.85, 'memory': 0.50, 'scaling': 0.70},
            'ssd': {'compute': 0.80, 'memory': 0.40, 'scaling': 0.80},
            'unet3d': {'compute': 0.90, 'memory': 0.60, 'scaling': 0.75},
            'rnnt': {'compute': 0.75, 'memory': 0.50, 'scaling': 0.70},
            'gnmt': {'compute': 0.70, 'memory': 0.60, 'scaling': 0.72},
            'transformer': {'compute': 0.75, 'memory': 0.65, 'scaling': 0.73},
            'ncf': {'compute': 0.30, 'memory': 0.80, 'scaling': 0.55},
            'minigo': {'compute': 0.60, 'memory': 0.40, 'scaling': 0.65},
            '70b_lora': {'compute': 0.85, 'memory': 0.80, 'scaling': 0.60},
            'dcnv2': {'compute': 0.85, 'memory': 0.50, 'scaling': 0.75},
            'diffusion': {'compute': 0.90, 'memory': 0.70, 'scaling': 0.70},
            'gnn': {'compute': 0.60, 'memory': 0.70, 'scaling': 0.65},
        }
        
        self.gpu_tdp = {
            'h100': 700, 'h200': 700, 'b200': 700,
            'a100': 400, 'v100': 300, 'l40s': 350, 'l40': 300,
            'a30': 165, 'a40': 300, 'rtx-4090': 450, '4090': 450,
            'l4': 72, 't4': 70, 'gaudi2': 600,
            'tpu-v4': 400, 'tpu-v5': 450, 'bow': 200, 'ipu': 200,
        }
    
    def get_gpu_score(self, gpu_model: str) -> float:
        if pd.isna(gpu_model):
            return 0.5
        
        gpu_str = str(gpu_model)
        gpu_lower = gpu_str.lower()
        
        for key, score in self.gpu_scores.items():
            if key in gpu_lower:
                return score
        
        patterns = {
            'b200|blackwell': 3.00,
            'h200': 2.50,
            'h100': 2.20,
            'a100': 1.00,
            'v100': 0.45,
            'l40s': 0.85,
            'a30': 0.55,
            '4090': 0.35,
            'l4': 0.25,
            'tpu.*v5': 2.00,
            'tpu.*v4': 1.20,
            'gaudi2': 0.90,
        }
        
        for pattern, score in patterns.items():
            if re.search(pattern, gpu_lower):
                return score
        
        return 0.5
    
    def get_gpu_tdp(self, gpu_model: str) -> float:
        if pd.isna(gpu_model):
            return 300
        
        gpu_lower = str(gpu_model).lower()
        
        for key, tdp in self.gpu_tdp.items():
            if key in gpu_lower:
                return tdp
        
        if 'h100' in gpu_lower or 'h200' in gpu_lower or 'b200' in gpu_lower:
            return 700
        elif 'a100' in gpu_lower:
            return 400
        elif 'v100' in gpu_lower:
            return 300
        
        return 300
    
    def get_benchmark_profile(self, benchmark: str) -> Dict[str, float]:
        bench_lower = str(benchmark).lower()
        
        if bench_lower in self.benchmark_profiles:
            return self.benchmark_profiles[bench_lower]
        
        return {'compute': 0.7, 'memory': 0.5, 'scaling': 0.7}


class DataPreprocessor:
    
    def __init__(self):
        self.kb = HardwareKnowledgeBase()
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Initial data size: {len(df)}")
        
        df = df[df['total_training_time'] > 0].copy()
        
        df['global_batch_size'] = df['global_batch_size'].fillna(128)
        df['train_samples'] = df['train_samples'].fillna(1000000)
        df['eval_samples'] = df['eval_samples'].fillna(50000)
        df['accelerator_model'] = df['accelerator_model'].fillna('unknown')
        df['framework'] = df['framework'].fillna('unknown')
        
        logger.info(f"Cleaned data size: {len(df)}")
        return df
    
    def extract_gpu_count(self, row: pd.Series) -> int:
        if 'total_accelerators' in row and pd.notna(row['total_accelerators']):
            count = int(row['total_accelerators'])
            if count > 0:
                return count
        
        if ('number_of_nodes' in row and pd.notna(row['number_of_nodes']) and
            'accelerators_per_node' in row and pd.notna(row['accelerators_per_node'])):
            return int(row['number_of_nodes']) * int(row['accelerators_per_node'])
        
        system = str(row.get('system', ''))
        patterns = [
            r'(\d+)x[a-z]\d+', r'x(\d+)', r'n(\d+)_',
            r'_n(\d+)', r'(\d+)gpu', r'(\d+)-gpu',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, system.lower())
            if match:
                count = int(match.group(1))
                if 1 <= count <= 4096:
                    return count
        
        return 1
    
    def compute_scaling_efficiency(self, gpu_count: int, base_scaling: float) -> float:
        if gpu_count <= 1:
            return 1.0
        elif gpu_count <= 8:
            return 0.95
        elif gpu_count <= 64:
            return base_scaling
        elif gpu_count <= 256:
            return base_scaling * 0.9
        else:
            return base_scaling * 0.8
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating features...")
        
        df = df.copy()
        
        df['gpu_count'] = df.apply(self.extract_gpu_count, axis=1)
        
        df['gpu_score'] = df['accelerator_model'].apply(self.kb.get_gpu_score)
        df['gpu_tdp'] = df['accelerator_model'].apply(self.kb.get_gpu_tdp)
        
        for benchmark in df['benchmark'].unique():
            profile = self.kb.get_benchmark_profile(benchmark)
            mask = df['benchmark'] == benchmark
            df.loc[mask, 'bench_compute'] = profile['compute']
            df.loc[mask, 'bench_memory'] = profile['memory']
            df.loc[mask, 'bench_scaling'] = profile['scaling']
        
        total = df['bench_compute'] + df['bench_memory']
        df['bench_compute_norm'] = df['bench_compute'] / total
        df['bench_memory_norm'] = df['bench_memory'] / total
        
        df['scaling_efficiency'] = df.apply(
            lambda row: self.compute_scaling_efficiency(
                row['gpu_count'], row['bench_scaling']
            ), axis=1
        )
        
        df['effective_compute'] = df['gpu_score'] * df['gpu_count']
        df['effective_compute_scaled'] = df['effective_compute'] * df['scaling_efficiency']
        
        df['batch_per_gpu'] = df['global_batch_size'] / df['gpu_count'].clip(lower=1)
        
        df['log_gpu_count'] = np.log1p(df['gpu_count'])
        df['log_batch_size'] = np.log1p(df['global_batch_size'])
        df['log_train_samples'] = np.log1p(df['train_samples'])
        df['log_batch_per_gpu'] = np.log1p(df['batch_per_gpu'])
        
        df['gpu_memory_ratio'] = df['batch_per_gpu'] / df['gpu_score']
        df['compute_memory_product'] = df['bench_compute'] * df['bench_memory']
        df['samples_per_gpu'] = df['train_samples'] / df['gpu_count'].clip(lower=1)
        df['log_samples_per_gpu'] = np.log1p(df['samples_per_gpu'])
        
        numeric_cols = ['train_samples', 'eval_samples', 'global_batch_size',
                       'number_of_nodes', 'accelerators_per_node']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df
    
    def create_text_embeddings(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating text embeddings (384 dimensions)...")
        
        texts = []
        for _, row in df.iterrows():
            text = (
                f"System: {row.get('system', 'unknown')}, "
                f"Benchmark: {row.get('benchmark', 'unknown')}, "
                f"Submitter: {row.get('submitter', 'unknown')}, "
                f"GPU: {row.get('accelerator_model', 'unknown')}, "
                f"Framework: {row.get('framework', 'unknown')}, "
                f"GPUs: {row.get('gpu_count', 1)}, "
                f"Nodes: {row.get('number_of_nodes', 1)}"
            )
            texts.append(text)
        
        embeddings = self.encoder.encode(texts, show_progress_bar=True)
        
        assert embeddings.shape[1] == 384
        
        embed_cols = [f'embed_{i}' for i in range(384)]
        embed_df = pd.DataFrame(embeddings, columns=embed_cols, index=df.index)
        
        return pd.concat([df, embed_df], axis=1)
    
    def add_faiss_features(self, df: pd.DataFrame, train_df: pd.DataFrame = None, 
                        k: int = 10, is_train: bool = True) -> pd.DataFrame:
        logger.info(f"Adding FAISS-based neighbor features (k={k}, is_train={is_train})...")
        
        embed_cols = [col for col in df.columns if col.startswith('embed_')]
        query_embeddings = df[embed_cols].values.astype('float32')
        
        if is_train:
            index_embeddings = query_embeddings
            reference_df = df
        else:
            if train_df is None:
                raise ValueError("Must provide train_df for test data")
            index_embeddings = train_df[embed_cols].values.astype('float32')
            reference_df = train_df
        
        dimension = index_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(index_embeddings)
        
        faiss_features = []
        
        for i in range(len(df)):
            if is_train:
                distances, indices = index.search(query_embeddings[i:i+1], k+1)
                similar_indices = indices[0][1:]
                similar_distances = distances[0][1:]
            else:
                distances, indices = index.search(query_embeddings[i:i+1], k)
                similar_indices = indices[0]
                similar_distances = distances[0]
            
            similar_gpu_counts = reference_df.iloc[similar_indices]['gpu_count'].values
            similar_gpu_scores = reference_df.iloc[similar_indices]['gpu_score'].values
            similar_batch_sizes = reference_df.iloc[similar_indices]['global_batch_size'].values
            similar_train_samples = reference_df.iloc[similar_indices]['train_samples'].values
            similar_effective_compute = reference_df.iloc[similar_indices]['effective_compute'].values
            
            epsilon = 1e-6
            weights = 1.0 / (similar_distances + epsilon)
            weights = weights / weights.sum()
            
            features = {
                'faiss_mean_distance': np.mean(similar_distances),
                'faiss_min_distance': np.min(similar_distances),
                'faiss_max_distance': np.max(similar_distances),
                'faiss_std_distance': np.std(similar_distances),
                'faiss_mean_gpu_count': np.mean(similar_gpu_counts),
                'faiss_mean_gpu_score': np.mean(similar_gpu_scores),
                'faiss_mean_batch_size': np.mean(similar_batch_sizes),
                'faiss_mean_train_samples': np.mean(similar_train_samples),
                'faiss_mean_effective_compute': np.mean(similar_effective_compute),
                'faiss_weighted_gpu_count': np.sum(weights * similar_gpu_counts),
                'faiss_weighted_effective_compute': np.sum(weights * similar_effective_compute),
            }
            
            faiss_features.append(features)
        
        faiss_df = pd.DataFrame(faiss_features, index=df.index)
        return pd.concat([df, faiss_df], axis=1)


def main():
    
    input_path = '/workspace/data/extracted/mlperf_results.csv'
    base_output_dir = '/workspace/data/processed'
    
    logger.info(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} rows")
    
    preprocessor = DataPreprocessor()
    
    df = preprocessor.clean_data(df)
    
    df_features = preprocessor.create_features(df)
    
    df_with_embeddings = preprocessor.create_text_embeddings(df_features)
    
    train_df, test_df = train_test_split(
        df_with_embeddings,
        test_size=0.2,
        random_state=42,
        stratify=df_with_embeddings['benchmark']
    )
    
    logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")
    
    train_df_final = preprocessor.add_faiss_features(train_df, k=10, is_train=True)
    test_df_final = preprocessor.add_faiss_features(
        test_df, train_df=train_df, k=10, is_train=False
    )
    
    numeric_features = [
        'gpu_count', 'gpu_score', 'gpu_tdp', 'scaling_efficiency',
        'effective_compute', 'effective_compute_scaled',
        'batch_per_gpu', 'log_gpu_count', 'log_batch_size', 'log_train_samples',
        'log_batch_per_gpu', 'log_samples_per_gpu',
        'train_samples', 'eval_samples', 'global_batch_size',
        'number_of_nodes', 'accelerators_per_node',
        'bench_compute', 'bench_memory', 'bench_scaling',
        'bench_compute_norm', 'bench_memory_norm',
        'gpu_memory_ratio', 'compute_memory_product',
        'samples_per_gpu'
    ]
    
    faiss_features = [col for col in train_df_final.columns if col.startswith('faiss_')]
    swmf_features = [col for col in train_df_final.columns if col.startswith('embed_')]
    
    logger.info("\n" + "="*70)
    logger.info("Saving 4 feature sets...")
    logger.info("="*70)
    
    output_dir_1 = os.path.join(base_output_dir, 'numeric_only')
    os.makedirs(output_dir_1, exist_ok=True)
    
    cols_1 = numeric_features + ['total_training_time', 'benchmark']
    train_df_final[cols_1].to_csv(os.path.join(output_dir_1, 'train.csv'), index=False)
    test_df_final[cols_1].to_csv(os.path.join(output_dir_1, 'test.csv'), index=False)
    
    with open(os.path.join(output_dir_1, 'feature_info.json'), 'w') as f:
        json.dump({
            'features': numeric_features,
            'n_features': len(numeric_features),
            'description': 'Numeric features only'
        }, f, indent=2)
    
    output_dir_2 = os.path.join(base_output_dir, 'swmf')
    os.makedirs(output_dir_2, exist_ok=True)
    
    cols_2 = swmf_features + ['total_training_time', 'benchmark']
    train_df_final[cols_2].to_csv(os.path.join(output_dir_2, 'train.csv'), index=False)
    test_df_final[cols_2].to_csv(os.path.join(output_dir_2, 'test.csv'), index=False)
    
    with open(os.path.join(output_dir_2, 'feature_info.json'), 'w') as f:
        json.dump({
            'features': swmf_features,
            'n_features': len(swmf_features),
            'description': 'SWMF embeddings only',
            'embedding_model': 'all-MiniLM-L6-v2'
        }, f, indent=2)
    
    output_dir_3 = os.path.join(base_output_dir, 'swmf_numeric')
    os.makedirs(output_dir_3, exist_ok=True)
    
    cols_3 = numeric_features + swmf_features + ['total_training_time', 'benchmark']
    train_df_final[cols_3].to_csv(os.path.join(output_dir_3, 'train.csv'), index=False)
    test_df_final[cols_3].to_csv(os.path.join(output_dir_3, 'test.csv'), index=False)
    
    with open(os.path.join(output_dir_3, 'feature_info.json'), 'w') as f:
        json.dump({
            'numeric_features': numeric_features,
            'swmf_features': swmf_features,
            'n_features': len(numeric_features) + len(swmf_features),
            'description': 'Numeric + SWMF embeddings',
            'embedding_model': 'all-MiniLM-L6-v2'
        }, f, indent=2)
    
    output_dir_4 = os.path.join(base_output_dir, 'swmf_numeric_faiss')
    os.makedirs(output_dir_4, exist_ok=True)
    
    cols_4 = numeric_features + swmf_features + faiss_features + ['total_training_time', 'benchmark']
    train_df_final[cols_4].to_csv(os.path.join(output_dir_4, 'train.csv'), index=False)
    test_df_final[cols_4].to_csv(os.path.join(output_dir_4, 'test.csv'), index=False)
    
    with open(os.path.join(output_dir_4, 'feature_info.json'), 'w') as f:
        json.dump({
            'numeric_features': numeric_features,
            'swmf_features': swmf_features,
            'faiss_features': faiss_features,
            'n_features': len(numeric_features) + len(swmf_features) + len(faiss_features),
            'description': 'Numeric + SWMF + FAISS neighbor features',
            'embedding_model': 'all-MiniLM-L6-v2',
            'faiss_config': {'k': 10, 'index_type': 'IndexFlatL2'}
        }, f, indent=2)
    
    dataset_info = {
        'dataset_name': 'MLCost v1.1',
        'total_samples': len(df),
        'train_samples': len(train_df_final),
        'test_samples': len(test_df_final),
        'number_of_benchmarks': df['benchmark'].nunique(),
        'number_of_hardware_models': df['accelerator_model'].nunique(),
    }
    
    for output_dir in [output_dir_1, output_dir_2, output_dir_3, output_dir_4]:
        with open(os.path.join(output_dir, 'dataset_info.json'), 'w') as f:
            json.dump(dataset_info, f, indent=2)
    
    print("\n" + "="*70)
    print("MLCost Preprocessing Complete")
    print("="*70)
    print(f"Total Samples: {dataset_info['total_samples']:,}")
    print(f"Train/Test Split: {len(train_df_final):,} / {len(test_df_final):,}")
    print(f"Benchmarks: {dataset_info['number_of_benchmarks']}")
    print(f"Hardware Models: {dataset_info['number_of_hardware_models']}")
    print("\nFeature Sets:")
    print(f"  1. numeric_only:      {len(numeric_features)}D")
    print(f"  2. swmf:              {len(swmf_features)}D")
    print(f"  3. swmf_numeric:      {len(numeric_features) + len(swmf_features)}D")
    print(f"  4. swmf_numeric_faiss: {len(numeric_features) + len(swmf_features) + len(faiss_features)}D")
    print(f"\nOutput directory: {base_output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()