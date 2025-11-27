#!/usr/bin/env python3

import os
import json
import re
import warnings
import numpy as np
import pandas as pd
import joblib
import faiss
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb

warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


@dataclass
class PredictionQuery:
    system: str = "user_query"
    benchmark: Optional[str] = None
    accelerator_model: Optional[str] = None
    gpu_count: int = 1
    global_batch_size: int = 128
    framework: str = "PyTorch"
    submitter: str = "user"
    number_of_nodes: int = 1
    train_samples: int = 1000000
    eval_samples: int = 50000
    accelerators_per_node: int = 8


@dataclass
class PredictionResult:
    time_seconds: float
    time_formatted: str
    predictions_by_model: Dict[str, float]
    similar_examples: List[Dict]
    cost_estimate: Dict
    confidence_interval: Optional[Tuple[float, float]] = None


class HardwareDatabase:
    
    GPU_SCORES = {
        'b200': 3.00, 'blackwell': 3.00, 'h200': 2.50, 'h100': 2.20,
        'tpu-v5': 2.00, 'tpu-v5e': 2.00, 'tpu-v4': 1.20,
        'a100': 1.00, 'gaudi2': 0.90, 'l40s': 0.85,
        'a30': 0.55, 'bow-ipu': 0.50, 'ipu': 0.50, 'v100': 0.45,
        'rtx-4090': 0.35, '4090': 0.35, 'l4': 0.25
    }
    
    BENCHMARK_PROFILES = {
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
    
    GPU_TDP = {
        'b200': 700, 'h200': 700, 'h100': 700, 'h100-pcie': 350,
        'a100': 400, 'a100-pcie': 300, 'v100': 300,
        'l40s': 350, 'l40': 300, 'a30': 165, 'a40': 300,
        'rtx-4090': 450, '4090': 450, 'l4': 72, 't4': 70,
        'gaudi2': 600, 'tpu-v4': 400, 'tpu-v5': 450,
        'bow-ipu': 200, 'ipu': 200,
    }
    
    ELECTRICITY_RATES = {
        'korea': 0.094,
        'asia_average': 0.110,
        'global_average': 0.155,
        'us_average': 0.147,
        'germany': 0.327,
        'us_california': 0.221,
        'us_texas': 0.089,
        'us_virginia': 0.091,
        'eu_average': 0.207,
        'eu_france': 0.174,
        'eu_netherlands': 0.216,
        'uk': 0.285,
        'china': 0.085,
        'japan': 0.157,
        'singapore': 0.172,
        'india': 0.092,
    }
    
    @classmethod
    def get_gpu_score(cls, gpu_model: str) -> float:
        if pd.isna(gpu_model):
            return 0.5
        gpu_clean = cls._clean_gpu_name(gpu_model)
        for key, score in cls.GPU_SCORES.items():
            if key.replace('-', '') in gpu_clean:
                return score
        return 0.5
    
    @classmethod
    def get_gpu_tdp(cls, gpu_model: str) -> float:
        if pd.isna(gpu_model):
            return 300
        gpu_clean = cls._clean_gpu_name(gpu_model)
        for key, tdp in cls.GPU_TDP.items():
            if key.replace('-', '') in gpu_clean:
                return tdp
        return 300
    
    @classmethod
    def get_benchmark_profile(cls, benchmark: str) -> Dict[str, float]:
        return cls.BENCHMARK_PROFILES.get(
            str(benchmark).lower(),
            {'compute': 0.7, 'memory': 0.5, 'scaling': 0.7}
        )
    
    @staticmethod
    def _clean_gpu_name(gpu_model: str) -> str:
        return str(gpu_model).lower().replace('-', '').replace('_', '').replace(' ', '')


class EnergyCalculator:
    """
    Energy and cost calculation based on paper Equation 3 and 4:
    E_total = P_device * n * U * O * t_h
    Cost = (E_total / 1000) * C_region
    """
    
    UTILIZATION = 0.80
    OVERHEAD = 1.30
    
    COMPARISON_REGIONS = ['korea', 'asia_average', 'global_average', 'us_average', 'germany']
    
    @classmethod
    def calculate(
        cls,
        gpu_model: str,
        gpu_count: int,
        training_hours: float
    ) -> Dict:
        tdp = HardwareDatabase.get_gpu_tdp(gpu_model)
        
        energy_wh = tdp * gpu_count * cls.UTILIZATION * cls.OVERHEAD * training_hours
        energy_kwh = energy_wh / 1000
        
        asia_rate = HardwareDatabase.ELECTRICITY_RATES['asia_average']
        
        regional_costs = {}
        for region in cls.COMPARISON_REGIONS:
            rate = HardwareDatabase.ELECTRICITY_RATES[region]
            cost = energy_kwh * rate
            ratio = rate / asia_rate
            regional_costs[region] = {
                'rate_per_kwh': rate,
                'cost_usd': cost,
                'ratio': ratio,
            }
        
        return {
            'tdp_watts': tdp,
            'gpu_count': gpu_count,
            'training_hours': training_hours,
            'utilization': cls.UTILIZATION,
            'overhead': cls.OVERHEAD,
            'energy_kwh': energy_kwh,
            'regional_costs': regional_costs,
        }


class FeatureEngineer:
    
    NUMERIC_FEATURES = [
        'gpu_count', 'gpu_score', 'gpu_tdp', 'scaling_efficiency',
        'effective_compute', 'effective_compute_scaled',
        'batch_per_gpu', 'log_gpu_count', 'log_batch_size', 'log_train_samples',
        'log_batch_per_gpu', 'log_samples_per_gpu',
        'train_samples', 'eval_samples', 'global_batch_size',
        'number_of_nodes', 'accelerators_per_node',
        'bench_compute', 'bench_memory', 'bench_scaling',
        'bench_compute_norm', 'bench_memory_norm',
        'gpu_memory_ratio', 'compute_memory_product', 'samples_per_gpu'
    ]
    
    FAISS_FEATURES = [
        'faiss_mean_distance', 'faiss_min_distance', 'faiss_max_distance',
        'faiss_std_distance', 'faiss_mean_gpu_count', 'faiss_mean_gpu_score',
        'faiss_mean_batch_size', 'faiss_mean_train_samples',
        'faiss_mean_effective_compute', 'faiss_weighted_gpu_count',
        'faiss_weighted_effective_compute'
    ]
    
    @staticmethod
    def compute_scaling_efficiency(gpu_count: int, base_scaling: float) -> float:
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
    
    @classmethod
    def create_numeric_features(cls, query: PredictionQuery) -> Dict[str, float]:
        features = {}
        
        features['gpu_count'] = query.gpu_count
        features['train_samples'] = query.train_samples
        features['eval_samples'] = query.eval_samples
        features['global_batch_size'] = query.global_batch_size
        features['number_of_nodes'] = query.number_of_nodes
        features['accelerators_per_node'] = min(8, query.gpu_count)
        
        features['gpu_score'] = HardwareDatabase.get_gpu_score(query.accelerator_model)
        features['gpu_tdp'] = HardwareDatabase.get_gpu_tdp(query.accelerator_model)
        
        profile = HardwareDatabase.get_benchmark_profile(query.benchmark)
        features['bench_compute'] = profile['compute']
        features['bench_memory'] = profile['memory']
        features['bench_scaling'] = profile['scaling']
        
        total_bench = features['bench_compute'] + features['bench_memory']
        features['bench_compute_norm'] = features['bench_compute'] / total_bench
        features['bench_memory_norm'] = features['bench_memory'] / total_bench
        
        features['scaling_efficiency'] = cls.compute_scaling_efficiency(
            query.gpu_count, profile['scaling']
        )
        
        features['effective_compute'] = features['gpu_score'] * query.gpu_count
        features['effective_compute_scaled'] = features['effective_compute'] * features['scaling_efficiency']
        
        features['batch_per_gpu'] = query.global_batch_size / max(1, query.gpu_count)
        
        features['log_gpu_count'] = np.log1p(query.gpu_count)
        features['log_batch_size'] = np.log1p(query.global_batch_size)
        features['log_train_samples'] = np.log1p(query.train_samples)
        features['log_batch_per_gpu'] = np.log1p(features['batch_per_gpu'])
        
        features['gpu_memory_ratio'] = features['batch_per_gpu'] / max(0.1, features['gpu_score'])
        features['compute_memory_product'] = features['bench_compute'] * features['bench_memory']
        features['samples_per_gpu'] = query.train_samples / max(1, query.gpu_count)
        features['log_samples_per_gpu'] = np.log1p(features['samples_per_gpu'])
        
        return features


class QueryParser:
    
    BENCHMARK_PATTERNS = {
        'bert': ['bert', 'language model', 'nlp'],
        'resnet': ['resnet', 'image', 'vision', 'imagenet'],
        'dlrm': ['dlrm', 'recommendation', 'recommender'],
        'gpt3': ['gpt3', 'gpt-3', 'large language model', 'llm'],
        'maskrcnn': ['mask', 'rcnn', 'detection', 'mask-rcnn'],
        'ssd': ['ssd', 'single shot', 'detector'],
        'unet3d': ['unet', '3d', 'medical', 'u-net'],
        'rnnt': ['rnnt', 'speech', 'asr', 'recurrent'],
        'gnmt': ['gnmt', 'translation', 'nmt'],
        'transformer': ['transformer', 'attention'],
        'ncf': ['ncf', 'neural collaborative'],
        'minigo': ['minigo', 'go', 'reinforcement'],
        'diffusion': ['diffusion', 'stable diffusion', 'imagen', 'dalle'],
        '70b_lora': ['70b', 'lora', 'llama'],
        'dcnv2': ['dcnv2', 'deformable'],
        'gnn': ['gnn', 'graph neural'],
    }
    
    GPU_PATTERNS = {
        'B200': ['b200', 'blackwell'],
        'H200': ['h200'],
        'H100': ['h100'],
        'A100': ['a100'],
        'V100': ['v100'],
        'L40S': ['l40s'],
        'L4': ['l4'],
        'A30': ['a30'],
        'RTX 4090': ['4090', 'rtx 4090'],
        'Gaudi2': ['gaudi2', 'gaudi 2'],
        'TPU v5': ['tpu v5', 'tpuv5', 'tpu-v5'],
        'TPU v4': ['tpu v4', 'tpuv4', 'tpu-v4'],
    }
    
    @classmethod
    def parse(cls, query: str) -> PredictionQuery:
        query_lower = query.lower()
        result = PredictionQuery()
        
        for benchmark, patterns in cls.BENCHMARK_PATTERNS.items():
            if any(p in query_lower for p in patterns):
                result.benchmark = benchmark
                break
        
        gpu_count_patterns = [
            r'(\d+)\s*x\s*(?:nvidia\s+)?([a-z]\d+)',
            r'(\d+)\s+(?:nvidia\s+)?([a-z]\d+)',
            r'on\s+(\d+)\s+(?:nvidia\s+)?([a-z]\d+)',
            r'(\d+)\s*(?:x|x)?\s*gpu',
            r'gpu[s]?\s*[:=]\s*(\d+)',
        ]
        
        for pattern in gpu_count_patterns:
            match = re.search(pattern, query_lower)
            if match:
                result.gpu_count = int(match.group(1))
                break
        
        for gpu_model, patterns in cls.GPU_PATTERNS.items():
            if any(p in query_lower for p in patterns):
                result.accelerator_model = gpu_model
                break
        
        batch_match = re.search(r'batch\s*(?:size)?\s*(?:of)?\s*(\d+)', query_lower)
        if batch_match:
            result.global_batch_size = int(batch_match.group(1))
        
        result.number_of_nodes = max(1, result.gpu_count // 8)
        result.accelerators_per_node = min(8, result.gpu_count)
        
        return result


class MLCostPredictor:
    
    ENSEMBLE_WEIGHTS = {'rf': 0.30, 'gb': 0.35, 'xgb': 0.35}
    
    MODEL_CONFIG = {
        'rf': {
            'class': RandomForestRegressor,
            'params': {'n_estimators': 200, 'max_depth': 20, 'random_state': 42, 'n_jobs': -1}
        },
        'gb': {
            'class': GradientBoostingRegressor,
            'params': {'n_estimators': 200, 'max_depth': 10, 'random_state': 42}
        },
        'xgb': {
            'class': xgb.XGBRegressor,
            'params': {'n_estimators': 200, 'max_depth': 10, 'random_state': 42, 'n_jobs': -1}
        }
    }
    
    def __init__(
        self,
        model_dir: str = './models',
        data_dir: str = './data/processed/swmf_numeric_faiss'
    ):
        self.model_dir = model_dir
        self.data_dir = data_dir
        
        import logging
        logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
        
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.models = {}
        self.scaler = None
        self.train_df = None
        self.test_df = None
        self.faiss_index = None
        self.feature_info = None
        self.training_metrics = None
        
        self._initialize()
    
    def _initialize(self):
        os.makedirs(self.model_dir, exist_ok=True)
        
        if self._models_exist():
            print("Loading pre-trained models...")
            self._load_models()
            self._load_data()
            self._build_faiss_index()
        else:
            print("No pre-trained models found. Training new models...")
            self._load_data()
            if self.train_df is not None:
                self._train_and_save_models()
                self._build_faiss_index()
            else:
                print("Warning: No training data available.")
    
    def _models_exist(self) -> bool:
        required_files = ['scaler.pkl', 'rf_model.pkl', 'gb_model.pkl', 'xgb_model.pkl']
        return all(os.path.exists(os.path.join(self.model_dir, f)) for f in required_files)
    
    def _load_models(self):
        self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.pkl'))
        
        for name in ['rf', 'gb', 'xgb']:
            path = os.path.join(self.model_dir, f'{name}_model.pkl')
            self.models[name] = joblib.load(path)
        
        metrics_path = os.path.join(self.model_dir, 'training_metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                self.training_metrics = json.load(f)
        
        print(f"Loaded {len(self.models)} models: {list(self.models.keys())}")
    
    def _load_data(self):
        train_path = os.path.join(self.data_dir, 'train.csv')
        test_path = os.path.join(self.data_dir, 'test.csv')
        
        if os.path.exists(train_path):
            self.train_df = pd.read_csv(train_path)
            print(f"Loaded training data: {len(self.train_df)} samples")
        
        if os.path.exists(test_path):
            self.test_df = pd.read_csv(test_path)
            print(f"Loaded test data: {len(self.test_df)} samples")
        
        feature_info_path = os.path.join(self.data_dir, 'feature_info.json')
        if os.path.exists(feature_info_path):
            with open(feature_info_path, 'r') as f:
                self.feature_info = json.load(f)
        else:
            self.feature_info = {
                'numeric_features': FeatureEngineer.NUMERIC_FEATURES,
                'swmf_features': [f'embed_{i}' for i in range(384)],
                'faiss_features': FeatureEngineer.FAISS_FEATURES
            }
    
    def _train_and_save_models(self):
        print("\n" + "="*60)
        print("Training MLCost Ensemble Models")
        print("="*60)
        
        feature_cols = (
            self.feature_info.get('numeric_features', []) +
            self.feature_info.get('swmf_features', []) +
            self.feature_info.get('faiss_features', [])
        )
        
        X_train = self.train_df[feature_cols].fillna(0).values
        y_train = np.log1p(self.train_df['total_training_time'].values)
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if self.test_df is not None:
            X_test = self.test_df[feature_cols].fillna(0).values
            y_test = np.log1p(self.test_df['total_training_time'].values)
            X_test_scaled = self.scaler.transform(X_test)
        
        self.training_metrics = {'individual': {}, 'ensemble': {}}
        predictions = {}
        
        for name, config in self.MODEL_CONFIG.items():
            print(f"\nTraining {name.upper()}...")
            
            model = config['class'](**config['params'])
            model.fit(X_train_scaled, y_train)
            self.models[name] = model
            
            if self.test_df is not None:
                y_pred_log = model.predict(X_test_scaled)
                predictions[name] = y_pred_log
                
                y_pred = np.expm1(y_pred_log)
                y_true = np.expm1(y_test)
                
                metrics = self._calculate_metrics(y_true, y_pred)
                self.training_metrics['individual'][name] = metrics
                
                print(f"  MAE: {metrics['mae']:.1f}s, RMSE: {metrics['rmse']:.1f}s, "
                      f"R2: {metrics['r2']:.4f}, MAPE: {metrics['mape']:.2f}%")
        
        if self.test_df is not None and predictions:
            print(f"\nEvaluating Ensemble (RF={self.ENSEMBLE_WEIGHTS['rf']}, "
                  f"GB={self.ENSEMBLE_WEIGHTS['gb']}, XGB={self.ENSEMBLE_WEIGHTS['xgb']})...")
            
            ensemble_pred_log = np.zeros(len(X_test_scaled))
            for name, weight in self.ENSEMBLE_WEIGHTS.items():
                ensemble_pred_log += predictions[name] * weight
            
            y_pred_ensemble = np.expm1(ensemble_pred_log)
            y_true = np.expm1(y_test)
            
            ensemble_metrics = self._calculate_metrics(y_true, y_pred_ensemble)
            self.training_metrics['ensemble'] = ensemble_metrics
            
            print(f"  MAE: {ensemble_metrics['mae']:.1f}s, RMSE: {ensemble_metrics['rmse']:.1f}s, "
                  f"R2: {ensemble_metrics['r2']:.4f}, MAPE: {ensemble_metrics['mape']:.2f}%")
        
        self._save_models()
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        mask = y_true > 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
        return {'mae': float(mae), 'rmse': float(rmse), 'r2': float(r2), 'mape': float(mape)}
    
    def _save_models(self):
        print(f"\nSaving models to {self.model_dir}...")
        
        joblib.dump(self.scaler, os.path.join(self.model_dir, 'scaler.pkl'))
        
        for name, model in self.models.items():
            joblib.dump(model, os.path.join(self.model_dir, f'{name}_model.pkl'))
        
        with open(os.path.join(self.model_dir, 'training_metrics.json'), 'w') as f:
            json.dump(self.training_metrics, f, indent=2)
        
        print("Models saved successfully!")
    
    def _build_faiss_index(self):
        if self.train_df is None:
            return
        
        embed_cols = [col for col in self.train_df.columns if col.startswith('embed_')]
        if not embed_cols:
            return
        
        embeddings = self.train_df[embed_cols].values.astype('float32')
        embeddings = np.ascontiguousarray(embeddings)
        
        self.faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
        self.faiss_index.add(embeddings)
        
        print(f"FAISS index built with {len(embeddings)} samples")
    
    def _create_text_embedding(self, query: PredictionQuery) -> np.ndarray:
        text = (
            f"System: {query.system}, "
            f"Benchmark: {query.benchmark}, "
            f"Submitter: {query.submitter}, "
            f"GPU: {query.accelerator_model}, "
            f"Framework: {query.framework}, "
            f"GPUs: {query.gpu_count}, "
            f"Nodes: {query.number_of_nodes}"
        )
        return self.encoder.encode([text], show_progress_bar=False)[0]
    
    def _get_faiss_features(self, embedding: np.ndarray, k: int = 10) -> Dict[str, float]:
        if self.faiss_index is None or self.train_df is None:
            return {feat: 0.0 for feat in FeatureEngineer.FAISS_FEATURES}
        
        embedding = np.ascontiguousarray(embedding.reshape(1, -1).astype('float32'))
        distances, indices = self.faiss_index.search(embedding, k)
        
        similar_rows = self.train_df.iloc[indices[0]]
        
        epsilon = 1e-6
        weights = 1.0 / (distances[0] + epsilon)
        weights = weights / weights.sum()
        
        return {
            'faiss_mean_distance': float(np.mean(distances[0])),
            'faiss_min_distance': float(np.min(distances[0])),
            'faiss_max_distance': float(np.max(distances[0])),
            'faiss_std_distance': float(np.std(distances[0])),
            'faiss_mean_gpu_count': float(similar_rows['gpu_count'].mean()),
            'faiss_mean_gpu_score': float(similar_rows['gpu_score'].mean()),
            'faiss_mean_batch_size': float(similar_rows['global_batch_size'].mean()),
            'faiss_mean_train_samples': float(similar_rows['train_samples'].mean()),
            'faiss_mean_effective_compute': float(similar_rows['effective_compute'].mean()),
            'faiss_weighted_gpu_count': float(np.sum(weights * similar_rows['gpu_count'].values)),
            'faiss_weighted_effective_compute': float(np.sum(weights * similar_rows['effective_compute'].values)),
        }
    
    def _create_feature_vector(self, query: PredictionQuery) -> np.ndarray:
        numeric_features = FeatureEngineer.create_numeric_features(query)
        embedding = self._create_text_embedding(query)
        faiss_features = self._get_faiss_features(embedding)
        
        feature_dict = {}
        feature_dict.update(numeric_features)
        
        for i, val in enumerate(embedding):
            feature_dict[f'embed_{i}'] = val
        
        feature_dict.update(faiss_features)
        
        all_feature_names = (
            self.feature_info.get('numeric_features', FeatureEngineer.NUMERIC_FEATURES) +
            self.feature_info.get('swmf_features', [f'embed_{i}' for i in range(384)]) +
            self.feature_info.get('faiss_features', FeatureEngineer.FAISS_FEATURES)
        )
        
        feature_array = [feature_dict.get(feat, 0) for feat in all_feature_names]
        return np.array(feature_array).reshape(1, -1)
    
    def _get_similar_examples(self, embedding: np.ndarray, k: int = 5) -> List[Dict]:
        if self.faiss_index is None or self.train_df is None:
            return []
        
        embedding = np.ascontiguousarray(embedding.reshape(1, -1).astype('float32'))
        distances, indices = self.faiss_index.search(embedding, k)
        
        examples = []
        for idx in indices[0]:
            row = self.train_df.iloc[idx]
            examples.append({
                'system': row.get('system', 'unknown'),
                'benchmark': row.get('benchmark', 'unknown'),
                'gpu_count': int(row['gpu_count']),
                'time': float(row['total_training_time']),
            })
        return examples
    
    def predict_time(self, query: PredictionQuery) -> Dict:
        if not self.models:
            raise RuntimeError("No trained models available. Please provide training data.")
        
        features = self._create_feature_vector(query)
        features_scaled = self.scaler.transform(features) if self.scaler else features
        
        predictions = {}
        for name, model in self.models.items():
            pred_log = model.predict(features_scaled)[0]
            predictions[name] = float(np.expm1(pred_log))
        
        ensemble_pred = sum(
            predictions[m] * self.ENSEMBLE_WEIGHTS.get(m, 1/len(predictions))
            for m in predictions
        )
        
        confidence_interval = None
        if len(predictions) > 1:
            pred_values = list(predictions.values())
            std_dev = np.std(pred_values)
            lower_bound = max(ensemble_pred * 0.5, ensemble_pred - 1.96 * std_dev)
            upper_bound = min(ensemble_pred * 2.0, ensemble_pred + 1.96 * std_dev)
            confidence_interval = (lower_bound, upper_bound)
        
        embedding = self._create_text_embedding(query)
        similar_examples = self._get_similar_examples(embedding)
        
        return {
            'prediction': ensemble_pred,
            'predictions_by_model': predictions,
            'similar_examples': similar_examples,
            'confidence_interval': confidence_interval
        }
    
    def predict_with_cost(
        self,
        query: Union[str, PredictionQuery]
    ) -> PredictionResult:
        if isinstance(query, str):
            query = QueryParser.parse(query)
        
        if not query.benchmark:
            raise ValueError("Benchmark must be specified")
        if not query.accelerator_model:
            raise ValueError("GPU model must be specified")
        
        time_result = self.predict_time(query)
        training_hours = time_result['prediction'] / 3600
        
        cost_estimate = EnergyCalculator.calculate(
            gpu_model=query.accelerator_model,
            gpu_count=query.gpu_count,
            training_hours=training_hours
        )
        
        return PredictionResult(
            time_seconds=time_result['prediction'],
            time_formatted=self._format_time(time_result['prediction']),
            predictions_by_model=time_result['predictions_by_model'],
            similar_examples=time_result['similar_examples'],
            cost_estimate=cost_estimate,
            confidence_interval=time_result.get('confidence_interval')
        )
    
    def get_model_metrics(self) -> Optional[Dict]:
        return self.training_metrics
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f} seconds"
        elif seconds < 3600:
            return f"{seconds/60:.1f} minutes"
        elif seconds < 86400:
            return f"{seconds/3600:.1f} hours"
        else:
            return f"{seconds/86400:.1f} days"


def print_available_options():
    print("\n" + "="*60)
    print("Available Benchmarks:")
    print("-"*60)
    benchmarks = list(HardwareDatabase.BENCHMARK_PROFILES.keys())
    for i, b in enumerate(benchmarks, 1):
        print(f"  {b:<12}", end="")
        if i % 4 == 0:
            print()
    print("\n")
    
    print("Available GPUs:")
    print("-"*60)
    gpus = list(QueryParser.GPU_PATTERNS.keys())
    for i, g in enumerate(gpus, 1):
        print(f"  {g:<12}", end="")
        if i % 4 == 0:
            print()
    print("\n")
    
    print("Regional Cost Comparison (automatically calculated):")
    print("-"*60)
    print(f"{'Region':<16} {'Rate ($/kWh)':<14}")
    print("-"*60)
    for region in EnergyCalculator.COMPARISON_REGIONS:
        rate = HardwareDatabase.ELECTRICITY_RATES[region]
        print(f"  {region:<16} ${rate:.3f}")
    print()


def interactive_mode(predictor: MLCostPredictor):
    print("\n" + "="*60)
    print("MLCost Interactive Mode")
    print("="*60)
    print("Enter natural language queries or type 'help' for options.")
    print("Type 'quit' or 'exit' to exit.\n")
    
    print("Example queries:")
    print("  - BERT training on 8 A100 GPUs")
    print("  - ResNet on 16 H100 GPUs with batch size 256")
    print("  - GPT3 training on 64 A100 GPUs")
    print()
    
    while True:
        try:
            user_input = input("Query> ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Exiting...")
                break
            
            if user_input.lower() == 'help':
                print_available_options()
                continue
            
            if user_input.lower() == 'metrics':
                metrics = predictor.get_model_metrics()
                if metrics:
                    print("\nModel Performance Metrics:")
                    print("-"*40)
                    if 'ensemble' in metrics:
                        m = metrics['ensemble']
                        print(f"Ensemble: MAE={m['mae']:.1f}s, R2={m['r2']:.4f}, MAPE={m['mape']:.2f}%")
                    for name, m in metrics.get('individual', {}).items():
                        print(f"  {name.upper()}: MAE={m['mae']:.1f}s, R2={m['r2']:.4f}")
                else:
                    print("No metrics available.")
                continue
            
            result = predictor.predict_with_cost(user_input)
            
            print(f"\n{'='*60}")
            print(f"Predicted Training Time: {result.time_formatted}")
            print(f"  ({result.time_seconds:.1f} seconds)")
            
            if result.confidence_interval:
                lower, upper = result.confidence_interval
                print(f"95% CI: {predictor._format_time(lower)} - {predictor._format_time(upper)}")
            
            print(f"\nModel Predictions:")
            for model, pred in result.predictions_by_model.items():
                print(f"  {model.upper()}: {predictor._format_time(pred)}")
            
            ce = result.cost_estimate
            print(f"\nEnergy Consumption: {ce['energy_kwh']:.3f} kWh")
            print(f"  (TDP: {ce['tdp_watts']}W, GPUs: {ce['gpu_count']}, "
                  f"U: {ce['utilization']}, O: {ce['overhead']})")
            
            print(f"\nRegional Cost Comparison:")
            print("-"*50)
            print(f"{'Region':<16} {'Rate ($/kWh)':<14} {'Cost ($)':<12} {'Ratio':<8}")
            print("-"*50)
            for region, data in ce['regional_costs'].items():
                print(f"{region:<16} {data['rate_per_kwh']:<14.3f} {data['cost_usd']:<12.3f} {data['ratio']:<8.2f}")
            print("-"*50)
            
            if result.similar_examples:
                print(f"\nSimilar Historical Runs:")
                for ex in result.similar_examples[:3]:
                    print(f"  - {ex['benchmark']}: {ex['gpu_count']} GPUs, "
                          f"{predictor._format_time(ex['time'])}")
            
            print()
            
        except ValueError as e:
            print(f"Error: {e}")
            print("Try 'help' to see available options.\n")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='MLCost: ML Training Cost Predictor')
    parser.add_argument('--model-dir', default='./models', help='Directory to save/load models')
    parser.add_argument('--data-dir', default='./data/processed/swmf_numeric_faiss', help='Data directory')
    parser.add_argument('--query', type=str, help='Single query to predict')
    parser.add_argument('--retrain', action='store_true', help='Force retrain models')
    
    args = parser.parse_args()
    
    if args.retrain:
        model_files = ['scaler.pkl', 'rf_model.pkl', 'gb_model.pkl', 'xgb_model.pkl']
        for f in model_files:
            path = os.path.join(args.model_dir, f)
            if os.path.exists(path):
                os.remove(path)
    
    predictor = MLCostPredictor(
        model_dir=args.model_dir,
        data_dir=args.data_dir
    )
    
    if args.query:
        result = predictor.predict_with_cost(args.query)
        
        print(f"\nQuery: {args.query}")
        print(f"Time: {result.time_formatted} ({result.time_seconds:.1f}s)")
        print(f"Energy: {result.cost_estimate['energy_kwh']:.3f} kWh")
        print(f"\nRegional Costs:")
        print(f"{'Region':<16} {'Rate':<10} {'Cost':<10} {'Ratio':<8}")
        for region, data in result.cost_estimate['regional_costs'].items():
            print(f"{region:<16} ${data['rate_per_kwh']:<9.3f} ${data['cost_usd']:<9.3f} {data['ratio']:.2f}")
    else:
        interactive_mode(predictor)


if __name__ == "__main__":
    main()