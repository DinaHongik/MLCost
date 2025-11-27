# MLCost

Predicting ML training time and electricity cost using MLPerf benchmark data.

## Performance

| Metric | Value |
|--------|-------|
| R² | 0.9770 |
| MAE | 227.9s |
| MAPE | 10.58% |

Trained on 17,611 MLPerf training records (v0.5–v4.1).

## Installation

```bash
git clone https://github.com/your-username/mlcost.git
cd mlcost
pip install -r requirements.txt
```

Or with Docker:

```bash
make build
make shell
```

## Data Preparation

Download and preprocess MLPerf data:

```bash
# 1. Clone MLPerf repositories
python scripts/downloader.py --config config/config.json

# 2. Parse logs
python scripts/extractor.py \
    --input data/raw \
    --output data/extracted/extracted_trdb.csv

# 3. Generate features
python mlcost/preprocess.py
```

## Usage

### CLI

```bash
# Single query
python mlcost/predictor.py --query "BERT on 8 A100 GPUs"

# Interactive mode
python mlcost/predictor.py
```

### Example Output

```
Query: BERT on 8 A100 GPUs
Time: 20.9 minutes (1255.7s)
Energy: 1.161 kWh

Regional Costs:
Region           Rate       Cost       Ratio
korea            $0.094     $0.109     0.85
asia_average     $0.110     $0.128     1.00
germany          $0.327     $0.380     2.97
```

### Python API

```python
from mlcost.predictor import MLCostPredictor

predictor = MLCostPredictor()
result = predictor.predict_with_cost("ResNet on 16 H100 GPUs")

print(f"Time: {result.time_formatted}")
print(f"Energy: {result.cost_estimate['energy_kwh']:.3f} kWh")
```

## Supported Benchmarks

BERT, ResNet, GPT3, DLRM, MaskRCNN, SSD, Unet3D, RNNT, Diffusion, GNN, etc.

## Supported GPUs

A100, H100, H200, V100, L40S, L4, RTX-4090, Gaudi2, TPU-v4, TPU-v5, etc.

## Cost Calculation

Based on Equations 3 and 4 from the paper:

```
E = TDP × GPU_count × 0.80 × 1.30 × time(h)
Cost = E(kWh) × regional_rate($/kWh)
```

Electricity rates from IEA/APERC 2023 data.

## Reproducing Paper Experiments

```bash
python mlcost/experiments.py
```

Generates LaTeX tables and figures in `experiments/`.

## Project Structure

```
mlcost/
├── mlcost/
│   ├── predictor.py      # Predictor + training
│   ├── preprocess.py     # Data preprocessing
│   └── experiments.py    # Paper experiments
├── scripts/
│   ├── downloader.py     # MLPerf download
│   └── extractor.py      # Log parsing
├── Dockerfile
├── Makefile
└── requirements.txt
```

## License

MIT