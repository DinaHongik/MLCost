#!/usr/bin/env python3

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MLCostExperiments:
    
    def __init__(self, data_dir='/workspace/data/processed/swmf_numeric_faiss'):
        self.data_dir = data_dir
        self.output_dir = Path('/workspace/experiments')
        self.results = {}
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'results').mkdir(exist_ok=True)
        (self.output_dir / 'tables').mkdir(exist_ok=True)
        (self.output_dir / 'plots').mkdir(exist_ok=True)
    
    def load_data(self):
        logger.info("Loading MLCost preprocessed data...")
        
        train_path = os.path.join(self.data_dir, 'train.csv')
        test_path = os.path.join(self.data_dir, 'test.csv')
        feature_info_path = os.path.join(self.data_dir, 'feature_info.json')
        
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        
        with open(feature_info_path, 'r') as f:
            feature_info = json.load(f)
        
        self.numeric_features = feature_info['numeric_features']
        self.swmf_features = feature_info['swmf_features']
        self.faiss_features = feature_info['faiss_features']
        
        self.feature_cols = self.numeric_features + self.swmf_features + self.faiss_features
        
        self.train_df['log_training_time'] = np.log1p(self.train_df['total_training_time'])
        self.test_df['log_training_time'] = np.log1p(self.test_df['total_training_time'])
        
        self.X_train = self.train_df[self.feature_cols].fillna(0).values
        self.y_train = self.train_df['log_training_time'].values
        
        self.X_test = self.test_df[self.feature_cols].fillna(0).values
        self.y_test = self.test_df['log_training_time'].values
        
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        logger.info(f"Train samples: {len(self.train_df)}")
        logger.info(f"Test samples: {len(self.test_df)}")
        logger.info(f"Features: {len(self.feature_cols)}D")
        logger.info(f"  - Numeric: {len(self.numeric_features)}D")
        logger.info(f"  - SWMF: {len(self.swmf_features)}D")
        logger.info(f"  - FAISS: {len(self.faiss_features)}D")
    
    def experiment_1_model_selection(self):
        logger.info("\n" + "="*70)
        logger.info("Experiment 1: Model Selection")
        logger.info("="*70)
        
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1, max_iter=2000),
            'SVR': SVR(kernel='rbf', C=1.0, gamma='scale'),
            'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=10, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
        }
        
        results = {}
        
        for name, model in models.items():
            logger.info(f"\nTraining {name}...")
            
            model.fit(self.X_train_scaled, self.y_train)
            
            y_pred_log = model.predict(self.X_test_scaled)
            y_pred = np.expm1(y_pred_log)
            y_true = np.expm1(self.y_test)
            
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            
            mask = y_true > 0
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else np.inf
            
            results[name] = {
                'mae': float(mae),
                'rmse': float(rmse),
                'r2': float(r2),
                'mape': float(mape)
            }
            
            logger.info(f"  MAE: {mae:.1f}s, RMSE: {rmse:.1f}s, R²: {r2:.4f}, MAPE: {mape:.1f}%")
        
        self.results['model_selection'] = results
        
        with open(self.output_dir / 'results' / 'model_selection.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def experiment_2_feature_analysis(self):
        logger.info("\n" + "="*70)
        logger.info("Experiment 2: Feature Analysis")
        logger.info("="*70)
        
        # Part A: Feature Ablation Study
        logger.info("\nPart A: Feature Ablation Study")
        
        feature_configs = {
            'Numeric Only (25D)': self.numeric_features,
            'SWMF (384D)': self.swmf_features,
            'Numeric + SWMF (409D)': self.numeric_features + self.swmf_features,
            'Numeric + SWMF + FAISS (420D)': self.feature_cols
        }
        
        models_to_test = {
            'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=10, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
        }
        
        ablation_results = {}
        
        for config_name, features in feature_configs.items():
            logger.info(f"\nTesting {config_name}...")
            ablation_results[config_name] = {}
            
            X_train_config = self.train_df[features].fillna(0).values
            X_test_config = self.test_df[features].fillna(0).values
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_config)
            X_test_scaled = scaler.transform(X_test_config)
            
            for model_name, model in models_to_test.items():
                model.fit(X_train_scaled, self.y_train)
                
                y_pred_log = model.predict(X_test_scaled)
                y_pred = np.expm1(y_pred_log)
                y_true = np.expm1(self.y_test)
                
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                r2 = r2_score(y_true, y_pred)
                
                mask = y_true > 0
                mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else np.inf
                
                ablation_results[config_name][model_name] = {
                    'mae': float(mae),
                    'rmse': float(rmse),
                    'r2': float(r2),
                    'mape': float(mape)
                }
                
                logger.info(f"  {model_name}: MAE={mae:.1f}s, R²={r2:.4f}")
        
        # Part B: Feature Importance Analysis
        logger.info("\nPart B: Feature Importance Analysis (420D)")
        
        importance_results = {}
        
        for model_name, model in models_to_test.items():
            logger.info(f"\nAnalyzing {model_name}...")
            
            model.fit(self.X_train_scaled, self.y_train)
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            else:
                continue
            
            # Get top 15 features
            feature_importance_pairs = list(zip(self.feature_cols, importances))
            sorted_features = sorted(feature_importance_pairs, key=lambda x: x[1], reverse=True)
            top_features = sorted_features[:15]
            
            # Calculate numeric vs embedding contribution
            numeric_importance = sum([imp for feat, imp in feature_importance_pairs 
                                     if feat in self.numeric_features])
            swmf_importance = sum([imp for feat, imp in feature_importance_pairs 
                                  if feat in self.swmf_features])
            faiss_importance = sum([imp for feat, imp in feature_importance_pairs 
                                   if feat in self.faiss_features])
            
            total_importance = numeric_importance + swmf_importance + faiss_importance
            
            importance_results[model_name] = {
                'top_features': [(feat, float(imp)) for feat, imp in top_features],
                'numeric_contribution': float(numeric_importance / total_importance * 100),
                'swmf_contribution': float(swmf_importance / total_importance * 100),
                'faiss_contribution': float(faiss_importance / total_importance * 100),
                'dominant_features': [feat for feat, imp in top_features[:3]]
            }
            
            logger.info(f"  Top 3 features: {importance_results[model_name]['dominant_features']}")
            logger.info(f"  Numeric: {importance_results[model_name]['numeric_contribution']:.1f}%")
            logger.info(f"  SWMF: {importance_results[model_name]['swmf_contribution']:.1f}%")
            logger.info(f"  FAISS: {importance_results[model_name]['faiss_contribution']:.1f}%")
        
        self.results['feature_analysis'] = {
            'ablation': ablation_results,
            'importance': importance_results
        }
        
        # Save results
        with open(self.output_dir / 'results' / 'feature_analysis.json', 'w') as f:
            json.dump(self.results['feature_analysis'], f, indent=2)
        
        return ablation_results, importance_results
    
    def experiment_3_benchmark_level_performance(self):
        logger.info("\n" + "="*70)
        logger.info("Experiment 2: Benchmark-Level Performance")
        logger.info("="*70)
        
        logger.info("Training model for benchmark analysis...")
        model = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
        model.fit(self.X_train_scaled, self.y_train)
        
        y_pred_log = model.predict(self.X_test_scaled)
        y_pred = np.expm1(y_pred_log)
        y_true = np.expm1(self.y_test)
        
        test_analysis = self.test_df.copy()
        test_analysis['predicted'] = y_pred
        test_analysis['actual'] = y_true
        test_analysis['error'] = np.abs(y_pred - y_true)
        test_analysis['percent_error'] = (test_analysis['error'] / y_true) * 100
        
        benchmark_results = {}
        
        for benchmark in test_analysis['benchmark'].unique():
            bench_df = test_analysis[test_analysis['benchmark'] == benchmark]
            
            if len(bench_df) == 0:
                continue
            
            benchmark_results[benchmark] = {
                'count': int(len(bench_df)),
                'mae': float(bench_df['error'].mean()),
                'mape': float(bench_df['percent_error'].mean()),
                'rmse': float(np.sqrt(np.mean(bench_df['error']**2))),
                'r2': float(r2_score(bench_df['actual'], bench_df['predicted'])) if len(bench_df) > 1 else 0.0
            }
        
        benchmark_results = dict(sorted(
            benchmark_results.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        ))
        
        logger.info("\nTop benchmarks by sample count:")
        for i, (bench, stats) in enumerate(list(benchmark_results.items())[:10]):
            logger.info(f"  {i+1}. {bench}: {stats['count']} samples, MAE={stats['mae']:.1f}s, MAPE={stats['mape']:.1f}%")
        
        self.results['benchmark_performance'] = benchmark_results
        
        with open(self.output_dir / 'results' / 'benchmark_performance.json', 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        
        return benchmark_results
    
    def experiment_4_cross_validation(self):
        logger.info("\n" + "="*70)
        logger.info("Experiment 4: Cross-Validation")
        logger.info("="*70)
        
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=10, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
        }
        
        weights = {
            'Random Forest': 0.30,
            'Gradient Boosting': 0.35,
            'XGBoost': 0.35
        }
        
        results = {}
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # Individual models CV
        for name, model in models.items():
            logger.info(f"\nCross-validating {name}...")
            
            fold_maes = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(self.X_train_scaled)):
                X_fold_train = self.X_train_scaled[train_idx]
                y_fold_train = self.y_train[train_idx]
                X_fold_val = self.X_train_scaled[val_idx]
                y_fold_val = self.y_train[val_idx]
                
                model.fit(X_fold_train, y_fold_train)
                
                y_pred_log = model.predict(X_fold_val)
                y_pred = np.expm1(y_pred_log)
                y_true = np.expm1(y_fold_val)
                
                mae = mean_absolute_error(y_true, y_pred)
                fold_maes.append(mae)
            
            results[name] = {
                'mean': float(np.mean(fold_maes)),
                'std': float(np.std(fold_maes)),
                'min': float(np.min(fold_maes)),
                'max': float(np.max(fold_maes)),
                'folds': [float(x) for x in fold_maes]
            }
            
            logger.info(f"  Mean MAE: {results[name]['mean']:.1f}s ± {results[name]['std']:.1f}s")
            logger.info(f"  Range: [{results[name]['min']:.1f}s, {results[name]['max']:.1f}s]")
        
        # Ensemble CV
        logger.info(f"\nCross-validating Ensemble...")
        ensemble_fold_maes = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(self.X_train_scaled)):
            X_fold_train = self.X_train_scaled[train_idx]
            y_fold_train = self.y_train[train_idx]
            X_fold_val = self.X_train_scaled[val_idx]
            y_fold_val = self.y_train[val_idx]
            
            # Train all models and get predictions
            fold_predictions = {}
            for name, model in models.items():
                model.fit(X_fold_train, y_fold_train)
                fold_predictions[name] = model.predict(X_fold_val)
            
            # Weighted ensemble prediction
            ensemble_pred_log = np.zeros(len(X_fold_val))
            for name, weight in weights.items():
                ensemble_pred_log += fold_predictions[name] * weight
            
            y_pred = np.expm1(ensemble_pred_log)
            y_true = np.expm1(y_fold_val)
            
            mae = mean_absolute_error(y_true, y_pred)
            ensemble_fold_maes.append(mae)
        
        results['Ensemble'] = {
            'mean': float(np.mean(ensemble_fold_maes)),
            'std': float(np.std(ensemble_fold_maes)),
            'min': float(np.min(ensemble_fold_maes)),
            'max': float(np.max(ensemble_fold_maes)),
            'folds': [float(x) for x in ensemble_fold_maes],
            'weights': weights
        }
        
        logger.info(f"  Mean MAE: {results['Ensemble']['mean']:.1f}s ± {results['Ensemble']['std']:.1f}s")
        logger.info(f"  Range: [{results['Ensemble']['min']:.1f}s, {results['Ensemble']['max']:.1f}s]")
        
        self.results['cross_validation'] = results
        
        with open(self.output_dir / 'results' / 'cross_validation.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def experiment_5_ensemble_evaluation(self):
        logger.info("\n" + "="*70)
        logger.info("Experiment 5: Ensemble Evaluation")
        logger.info("="*70)
        
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=10, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
        }
        
        weights = {
            'Random Forest': 0.30,
            'Gradient Boosting': 0.35,
            'XGBoost': 0.35
        }
        
        results = {}
        predictions = {}
        
        for name, model in models.items():
            logger.info(f"\nTraining {name}...")
            
            model.fit(self.X_train_scaled, self.y_train)
            
            y_pred_log = model.predict(self.X_test_scaled)
            predictions[name] = y_pred_log
            
            y_pred = np.expm1(y_pred_log)
            y_true = np.expm1(self.y_test)
            
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            
            mask = y_true > 0
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else np.inf
            
            results[name] = {
                'mae': float(mae),
                'rmse': float(rmse),
                'r2': float(r2),
                'mape': float(mape)
            }
            
            logger.info(f"  MAE: {mae:.1f}s, RMSE: {rmse:.1f}s, R²: {r2:.4f}, MAPE: {mape:.1f}%")
        
        logger.info("\nComputing ensemble predictions...")
        ensemble_pred_log = np.zeros(len(self.X_test_scaled))
        for name, weight in weights.items():
            ensemble_pred_log += predictions[name] * weight
        
        y_pred_ensemble = np.expm1(ensemble_pred_log)
        y_true = np.expm1(self.y_test)
        
        mae = mean_absolute_error(y_true, y_pred_ensemble)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred_ensemble))
        r2 = r2_score(y_true, y_pred_ensemble)
        
        mask = y_true > 0
        mape = np.mean(np.abs((y_true[mask] - y_pred_ensemble[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else np.inf
        
        results['Ensemble'] = {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'mape': float(mape),
            'weights': weights
        }
        
        logger.info(f"\nEnsemble Performance:")
        logger.info(f"  MAE: {mae:.1f}s, RMSE: {rmse:.1f}s, R²: {r2:.4f}, MAPE: {mape:.1f}%")
        
        self.results['ensemble_evaluation'] = results
        
        with open(self.output_dir / 'results' / 'ensemble_evaluation.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def save_results(self):
        logger.info("\nSaving summary results...")
        
        summary = {
            'experiment_1_model_selection': self.results.get('model_selection', {}),
            'experiment_2_feature_analysis': self.results.get('feature_analysis', {}),
            'experiment_3_benchmark_performance': self.results.get('benchmark_performance', {}),
            'experiment_4_cross_validation': self.results.get('cross_validation', {}),
            'experiment_5_ensemble_evaluation': self.results.get('ensemble_evaluation', {})
        }
        
        with open(self.output_dir / 'results' / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Results saved to {self.output_dir / 'results'}")
    
    def generate_tables(self):
        logger.info("\nGenerating LaTeX tables...")
        
        tables_dir = self.output_dir / 'tables'
        
        if 'model_selection' in self.results:
            self._generate_table1(tables_dir)
        
        if 'feature_analysis' in self.results:
            self._generate_table2(tables_dir)
        
        if 'benchmark_performance' in self.results:
            self._generate_table3_4(tables_dir)
        
        if 'cross_validation' in self.results:
            self._generate_table5(tables_dir)
        
        if 'ensemble_evaluation' in self.results:
            self._generate_table6(tables_dir)
        
        logger.info(f"Tables saved to {tables_dir}")
    
    def _generate_table1(self, tables_dir):
        results = self.results['model_selection']
        
        with open(tables_dir / 'table1_model_comparison.tex', 'w') as f:
            f.write("\\begin{table}[t]\n")
            f.write("\\centering\n")
            f.write("\\caption{Prediction Performance of Regression Models.}\n")
            f.write("\\label{tab:model_comparison}\n")
            f.write("\\begin{tabular}{l S[table-format=4.1] S[table-format=4.1] S[table-format=1.4] S[table-format=2.1]}\n")
            f.write("\\toprule\n")
            f.write("\\textbf{Model} & \\textbf{MAE (s)} & \\textbf{RMSE (s)} & {\\boldmath$R^2$} & \\textbf{MAPE (\\%)} \\\\\n")
            f.write("\\midrule\n")
            
            sorted_results = sorted(results.items(), key=lambda x: x[1]['mae'])
            
            for name, metrics in sorted_results:
                f.write(f"{name} & {metrics['mae']:.1f} & {metrics['rmse']:.1f} & {metrics['r2']:.4f} & {metrics['mape']:.1f} \\\\\n")
            
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
    
    def _generate_table2(self, tables_dir):
        results = self.results['feature_analysis']['ablation']
        
        with open(tables_dir / 'table2_feature_ablation.tex', 'w') as f:
            f.write("\\begin{table}[t]\n")
            f.write("\\centering\n")
            f.write("\\caption{Feature ablation study results.}\n")
            f.write("\\label{tab:feature_ablation}\n")
            f.write("\\begin{tabular}{L{4cm} l S[table-format=3.1] S[table-format=1.4]}\n")
            f.write("\\toprule\n")
            f.write("\\textbf{Feature Set} & \\textbf{Model} & \\textbf{MAE (s)} & {\\boldmath$R^2$} \\\\\n")
            f.write("\\midrule\n")
            
            for config, models in results.items():
                first_model = True
                for model_name in ['Random Forest', 'Gradient Boosting', 'XGBoost']:
                    if model_name in models:
                        metrics = models[model_name]
                        config_str = config if first_model else ""
                        f.write(f"{config_str} & {model_name} & {metrics['mae']:.1f} & {metrics['r2']:.4f} \\\\\n")
                        first_model = False
                if not first_model:
                    f.write("\\addlinespace[0.3em]\n")
            
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
    
    def _generate_table3_4(self, tables_dir):
        results = self.results['benchmark_performance']
        
        major_benchmarks = ['unet3d', 'bert', 'resnet', 'rnnt', 'ssd', 'maskrcnn']
        specialized_benchmarks = ['70b_lora', 'dcnv2', 'minigo', 'diffusion', 'dlrm', 'gnn']
        
        with open(tables_dir / 'table3_major_benchmarks.tex', 'w') as f:
            f.write("\\begin{table}[t]\n")
            f.write("\\centering\n")
            f.write("\\caption{Prediction metrics for major benchmarks.}\n")
            f.write("\\label{tab:benchmark_major}\n")
            f.write("\\begin{tabular}{L{2.8cm} S[table-format=4.1] S[table-format=2.1]}\n")
            f.write("\\toprule\n")
            f.write("\\textbf{Benchmark} & \\textbf{MAE (s)} & \\textbf{MAPE (\\%)} \\\\\n")
            f.write("\\midrule\n")
            
            for bench in major_benchmarks:
                if bench in results:
                    metrics = results[bench]
                    bench_name = bench.replace('_', ' ').title()
                    f.write(f"{bench_name} & {metrics['mae']:.1f} & {metrics['mape']:.1f} \\\\\n")
            
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
        
        with open(tables_dir / 'table4_specialized_benchmarks.tex', 'w') as f:
            f.write("\\begin{table}[t]\n")
            f.write("\\centering\n")
            f.write("\\caption{Prediction metrics for specialized benchmarks.}\n")
            f.write("\\label{tab:benchmark_special}\n")
            f.write("\\begin{tabular}{L{2.8cm} S[table-format=4.1] S[table-format=2.1]}\n")
            f.write("\\toprule\n")
            f.write("\\textbf{Benchmark} & \\textbf{MAE (s)} & \\textbf{MAPE (\\%)} \\\\\n")
            f.write("\\midrule\n")
            
            for bench in specialized_benchmarks:
                if bench in results:
                    metrics = results[bench]
                    bench_name = bench.replace('_', ' ').title()
                    f.write(f"{bench_name} & {metrics['mae']:.1f} & {metrics['mape']:.1f} \\\\\n")
            
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
    
    def _generate_table5(self, tables_dir):
        results = self.results['cross_validation']
        
        with open(tables_dir / 'table5_cross_validation.tex', 'w') as f:
            f.write("\\begin{table}[t]\n")
            f.write("\\centering\n")
            f.write("\\caption{5-fold cross-validation results.}\n")
            f.write("\\label{tab:cross_validation}\n")
            f.write("\\begin{tabular}{l S[table-format=3.1] S[table-format=3.1] S[table-format=3.1] S[table-format=3.1]}\n")
            f.write("\\toprule\n")
            f.write("\\textbf{Model} & \\textbf{Mean (s)} & \\textbf{Std (s)} & \\textbf{Min (s)} & \\textbf{Max (s)}\\\\\n")
            f.write("\\midrule\n")
            
            for name, metrics in results.items():
                f.write(f"{name} & {metrics['mean']:.1f} & {metrics['std']:.1f} & {metrics['min']:.1f} & {metrics['max']:.1f} \\\\\n")
            
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
    
    def _generate_table6(self, tables_dir):
        results = self.results['ensemble_evaluation']
        
        with open(tables_dir / 'table6_ensemble.tex', 'w') as f:
            f.write("\\begin{table}[t]\n")
            f.write("\\centering\n")
            f.write("\\caption{Ensemble vs. major regression models.}\n")
            f.write("\\label{tab:ensemble_comparison}\n")
            f.write("\\begin{tabular}{l S[table-format=3.1] S[table-format=4.1] S[table-format=1.4] S[table-format=2.2]}\n")
            f.write("\\toprule\n")
            f.write("\\textbf{Model} & \\textbf{MAE} & \\textbf{RMSE} & {\\boldmath$R^2$} & \\textbf{MAPE} \\\\\n")
            f.write("\\midrule\n")
            
            for name in ['Gradient Boosting', 'XGBoost', 'Random Forest']:
                if name in results:
                    metrics = results[name]
                    f.write(f"{name} & {metrics['mae']:.1f} & {metrics['rmse']:.1f} & {metrics['r2']:.4f} & {metrics['mape']:.2f} \\\\\n")
            
            if 'Ensemble' in results:
                metrics = results['Ensemble']
                f.write(f"\\textbf{{Ensemble}} & {metrics['mae']:.1f} & {metrics['rmse']:.1f} & {metrics['r2']:.4f} & {metrics['mape']:.2f} \\\\\n")
            
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
    
    def create_plots(self):
        logger.info("\nCreating plots...")
        
        plots_dir = self.output_dir / 'plots'
        
        if 'model_selection' in self.results:
            self._plot_model_comparison(plots_dir)
        
        if 'benchmark_performance' in self.results:
            self._plot_benchmark_performance(plots_dir)
        
        logger.info(f"Plots saved to {plots_dir}")
    
    def _plot_model_comparison(self, plots_dir):
        results = self.results['model_selection']
        
        models = list(results.keys())
        mae_values = [results[m]['mae'] for m in models]
        
        plt.figure(figsize=(10, 6))
        plt.barh(models, mae_values)
        plt.xlabel('MAE (seconds)')
        plt.title('Model Comparison - Mean Absolute Error')
        plt.tight_layout()
        plt.savefig(plots_dir / 'figure1_model_comparison.pdf')
        plt.close()
    
    def _plot_benchmark_performance(self, plots_dir):
        results = self.results['benchmark_performance']
        
        benchmarks = list(results.keys())[:15]
        mae_values = [results[b]['mae'] for b in benchmarks]
        mape_values = [results[b]['mape'] for b in benchmarks]
        counts = [results[b]['count'] for b in benchmarks]
        
        # Pastel color mapping based on MAPE
        colors = []
        for mape in mape_values:
            if mape < 5:
                colors.append('#a8e6cf')  # Pastel green - Excellent
            elif mape < 10:
                colors.append('#ffd3b6')  # Pastel orange - Good
            else:
                colors.append('#ffaaa5')  # Pastel red - Challenging
        
        fig, ax = plt.subplots(figsize=(14, 9))
        
        # Plot with pastel colors
        scatter = ax.scatter(mae_values, mape_values, 
                            s=[c*8 for c in counts], 
                            c=colors,
                            alpha=0.8,
                            edgecolors='white',
                            linewidths=2)
        
        # Add labels
        for i, bench in enumerate(benchmarks):
            ax.annotate(bench, 
                       (mae_values[i], mape_values[i]), 
                       fontsize=10,
                       fontweight='bold',
                       ha='center',
                       va='center')
        
        ax.set_xlabel('MAE (seconds)', fontsize=13, fontweight='bold')
        ax.set_ylabel('MAPE (%)', fontsize=13, fontweight='bold')
        ax.set_title('Benchmark-Level Prediction Performance\n(bubble size = sample count)', 
                    fontsize=15, fontweight='bold', pad=20)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#a8e6cf', label='Excellent (MAPE < 5%)'),
            Patch(facecolor='#ffd3b6', label='Good (5% ≤ MAPE < 10%)'),
            Patch(facecolor='#ffaaa5', label='Challenging (MAPE ≥ 10%)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.95)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'figure4_benchmark_performance.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(plots_dir / 'figure4_benchmark_performance.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    experiments = MLCostExperiments(
        data_dir='/workspace/data/processed/swmf_numeric_faiss'
    )
    
    experiments.load_data()
    
    print("\n" + "="*70)
    print("MLCost Experiments")
    print("="*70)
    
    exp1_results = experiments.experiment_1_model_selection()
    
    exp2_ablation, exp2_importance = experiments.experiment_2_feature_analysis()
    
    exp3_results = experiments.experiment_3_benchmark_level_performance()
    
    exp4_results = experiments.experiment_4_cross_validation()
    
    exp5_results = experiments.experiment_5_ensemble_evaluation()
    
    experiments.save_results()
    experiments.generate_tables()
    experiments.create_plots()
    
    print("\n" + "="*70)
    print("All experiments completed!")
    print("Results saved to /workspace/experiments/")
    print("="*70)


if __name__ == "__main__":
    main()