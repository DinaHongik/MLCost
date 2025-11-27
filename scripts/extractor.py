"""
MLPerf Training Results Extractor
"""

import os
import re
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("mlperf-extractor")


class MLPerfExtractor:
    def __init__(self):
        self.records: List[Dict[str, Any]] = []
        self.systems_seen: set[str] = set()
        self.benchmarks_seen: set[str] = set()

    def extract_data(self, raw_data_dir: str) -> None:
        logger.info(f"Starting extraction from: {raw_data_dir}")
        self.records.clear()
        self.systems_seen.clear()
        self.benchmarks_seen.clear()
        
        version_dirs = self._find_version_directories(raw_data_dir)
        logger.info(f"Found {len(version_dirs)} version directories")
        
        for version_path in version_dirs:
            self._process_version_directory(version_path)
        
        logger.info(
            f"Extraction complete: {len(self.records)} records, "
            f"{len(self.systems_seen)} systems, "
            f"{len(self.benchmarks_seen)} benchmarks"
        )
    
    def _find_version_directories(self, root_dir: str) -> List[Path]:
        version_pattern = re.compile(r"^(training_)?results_v\d+\.\d+$")
        version_dirs = []
        
        for item in Path(root_dir).iterdir():
            if item.is_dir() and version_pattern.match(item.name):
                version_dirs.append(item)
        
        return sorted(version_dirs)
    
    def _process_version_directory(self, version_path: Path) -> None:
        version_name = version_path.name
        logger.info(f"Processing version: {version_name}")
        
        for submitter_path in version_path.iterdir():
            if not submitter_path.is_dir() or submitter_path.name.startswith('.'):
                continue
            
            submitter_name = submitter_path.name
            results_dir = submitter_path / "results"
            systems_dir = submitter_path / "systems"
            
            if not results_dir.exists() or not systems_dir.exists():
                logger.warning(f"Missing results or systems directory for {submitter_name}")
                continue
            
            system_configs = self._load_system_configs(systems_dir)
            
            for system_dir in results_dir.iterdir():
                if not system_dir.is_dir():
                    continue
                
                system_name = system_dir.name
                system_info = system_configs.get(system_name, {})
                
                if not system_info:
                    logger.warning(f"No system info found for: {system_name}")
                
                self.systems_seen.add(system_name)
                
                self._process_system_benchmarks(
                    system_dir, version_name, submitter_name, 
                    system_name, system_info
                )
    
    def _load_system_configs(self, systems_dir: Path) -> Dict[str, Dict[str, Any]]:
        configs = {}
        
        for json_file in systems_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    if not content.strip():
                        logger.warning(f"Empty JSON file: {json_file}")
                        continue
                    
                    content = self._fix_json_content(content)
                    data = json.loads(content)
                    system_name = json_file.stem
                    configs[system_name] = data
            except Exception as e:
                logger.error(f"Failed to load system config {json_file}: {e}")
        
        return configs
    
    def _fix_json_content(self, content: str) -> str:
        content = re.sub(r',\s*}', '}', content)
        content = re.sub(r',\s*]', ']', content)
        return content
    
    def _process_system_benchmarks(
        self, 
        system_dir: Path, 
        version: str, 
        submitter: str, 
        system: str, 
        system_info: Dict[str, Any]
    ) -> None:
        for benchmark_dir in system_dir.iterdir():
            if not benchmark_dir.is_dir():
                continue
            
            benchmark_name = benchmark_dir.name
            
            if any(x in benchmark_name.lower() for x in ['compliance', 'reference']):
                continue
            
            if '_' in benchmark_name:
                benchmark_name = benchmark_name.split('_', 1)[-1]
            
            self.benchmarks_seen.add(benchmark_name)
            
            log_patterns = ['*.log', '*.out', '*.txt', '*.stdout', '*.stderr']
            result_files = []
            
            for pattern in log_patterns:
                result_files.extend(list(benchmark_dir.glob(f"**/{pattern}")))
            
            seen = set()
            unique_files = []
            for f in result_files:
                if f not in seen:
                    seen.add(f)
                    unique_files.append(f)
            
            logger.debug(f"Found {len(unique_files)} log files in {benchmark_dir.name}")
            
            if not unique_files:
                logger.warning(f"No log files found in {benchmark_dir}")
            
            for result_file in unique_files:
                self._process_result_file(
                    result_file, version, submitter, 
                    system, system_info, benchmark_name
                )
    
    def _process_result_file(
        self,
        result_path: Path,
        version: str,
        submitter: str,
        system: str,
        system_info: Dict[str, Any],
        benchmark: str
    ) -> None:
        logger.debug(f"Processing file: {result_path}")
        
        record = {
            'version': version,
            'submitter': submitter,
            'system': system,
            'benchmark': benchmark,
            'category': self._get_benchmark_category(benchmark),
            'system_name': system_info.get('system_name', system),
            'accelerator_model': system_info.get('accelerator_model_name'),
            'host_processor': system_info.get('host_processor_model_name'),
            'framework': system_info.get('framework'),
            'number_of_nodes': self._safe_get_int(system_info, 'number_of_nodes', 1),
            'accelerators_per_node': self._safe_get_int(system_info, 'accelerators_per_node', 1),
        }
        
        record['total_accelerators'] = record['number_of_nodes'] * record['accelerators_per_node']
        
        metrics = self._extract_metrics_from_log(result_path)
        record.update(metrics)
        
        if any(k in record for k in ['eval_accuracy', 'total_training_time']):
            self.records.append(record)
        else:
            logger.debug(f"Skipping record without key metrics: {result_path}")
    
    def _safe_get_int(self, data: Dict[str, Any], key: str, default: int) -> int:
        try:
            return int(data.get(key, default))
        except (TypeError, ValueError):
            logger.debug(f"Invalid {key}, using default: {default}")
            return default
    
    def _safe_get_numeric(self, value: Any) -> Optional[float]:
        """Safely convert value to numeric, handling lists and other types"""
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, list) and value:
            return self._safe_get_numeric(value[0])
        elif isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return None
        return None
    
    def _extract_metrics_from_log(self, log_path: Path) -> Dict[str, Any]:
        metrics = {}
        run_start = None
        run_stop = None
        
        try:
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if ':::MLLOG' not in line:
                        continue
                    
                    try:
                        json_str = line.split(':::MLLOG', 1)[1].strip()
                        entry = json.loads(json_str)
                        
                        key = entry.get('key')
                        value = entry.get('value')
                        
                        if key in ['train_samples', 'eval_samples', 'global_batch_size']:
                            numeric_value = self._safe_get_numeric(value)
                            if numeric_value is not None:
                                metrics[key] = numeric_value
                        
                        elif key == 'eval_accuracy':
                            metrics[key] = value
                        
                        elif key == 'run_start':
                            time_ms = entry.get('time_ms', value)
                            time_numeric = self._safe_get_numeric(time_ms)
                            if time_numeric is not None:
                                run_start = time_numeric / 1000.0
                        
                        elif key == 'run_stop':
                            time_ms = entry.get('time_ms', value)
                            time_numeric = self._safe_get_numeric(time_ms)
                            if time_numeric is not None:
                                run_stop = time_numeric / 1000.0
                        
                        elif key == 'tracked_stats' and isinstance(value, dict):
                            if 'throughput' in value:
                                throughput = self._safe_get_numeric(value['throughput'])
                                if throughput is not None:
                                    metrics['samples_per_second'] = throughput
                    
                    except (json.JSONDecodeError, IndexError):
                        continue
            
            if run_start is not None and run_stop is not None:
                run_start, run_stop = self._normalize_timestamps(run_start, run_stop)
                
                if run_start and run_stop and run_stop > run_start:
                    metrics['total_training_time'] = run_stop - run_start
                    
                    if 'samples_per_second' not in metrics and 'train_samples' in metrics:
                        if metrics['total_training_time'] > 0:
                            metrics['samples_per_second'] = (
                                metrics['train_samples'] / metrics['total_training_time']
                            )
        
        except Exception as e:
            logger.error(f"Failed to parse log file {log_path}: {e}")
        
        return metrics
    
    def _normalize_timestamps(self, start: float, stop: float) -> tuple:
        return start, stop
    
    def _get_benchmark_category(self, benchmark: str) -> str:
        benchmark_lower = benchmark.lower()
        
        categories = {
            'nlp': ['bert', 'gpt', 'transformer'],
            'vision': ['resnet', 'maskrcnn', 'ssd', 'unet'],
            'recommendation': ['dlrm', 'recommendation'],
            'speech': ['rnnt', 'speech'],
            'reinforcement': ['minigo', 'reinforcement']
        }
        
        for category, keywords in categories.items():
            if any(keyword in benchmark_lower for keyword in keywords):
                return category
        
        return 'other'
    
    def export_to_csv(self, output_path: str) -> pd.DataFrame:
        if not self.records:
            logger.warning("No records to export")
            return pd.DataFrame()
        
        df = pd.DataFrame(self.records)
        
        sort_columns = ['version', 'submitter', 'system', 'benchmark']
        df = df.sort_values(by=sort_columns)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(df)} records to {output_path}")
        
        logger.info(f"Summary: {df['benchmark'].nunique()} unique benchmarks")
        logger.info(f"Summary: {df['system'].nunique()} unique systems")
        logger.info(f"Summary: {df['submitter'].nunique()} unique submitters")
        
        return df

    def export_to_jsonl(self, output_path: str):
        with open(output_path, 'w', encoding='utf-8') as f:
            for rec in self.records:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description="Extract MLPerf training results")
    parser.add_argument('--input', required=True, help='Root directory containing raw MLPerf data')
    parser.add_argument('--output', required=True, help='Output file path (.csv or .jsonl)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    output_ext = Path(args.output).suffix.lower()
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.isdir(args.input):
        logger.error(f"Input directory does not exist: {args.input}")
        return 1

    try:
        extractor = MLPerfExtractor()
        extractor.extract_data(args.input)

        if output_ext == '.jsonl':
            extractor.export_to_jsonl(args.output)
        elif output_ext == '.csv':
            extractor.export_to_csv(args.output)
        else:
            logger.error(f"Unsupported output format: {output_ext}")
            return 1

        return 0

    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())



# python extractor.py --input /workspace/mlperf_data/raw --output /workspace/cost_v3/data/extracted/extracted_trdb.csv



# 2025-06-19 06:05:04,988 [INFO] mlperf-extractor: Extraction complete: 17645 records, 693 systems, 16 benchmarks
# 2025-06-19 06:05:05,272 [INFO] mlperf-extractor: Exported 17645 records to /workspace/cost_v3/data/extracted/extracted_trdb.csv
# 2025-06-19 06:05:05,273 [INFO] mlperf-extractor: Summary: 15 unique benchmarks
# 2025-06-19 06:05:05,274 [INFO] mlperf-extractor: Summary: 659 unique systems
# 2025-06-19 06:05:05,276 [INFO] mlperf-extractor: Summary: 51 unique submitters