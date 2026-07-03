# TPVEPR

TPVEPR, short for Tissue-specific Promoter Variant eQTL Prioritization Ranker,
is a relation-aware set Transformer for ranking promoter expression
quantitative trait locus (eQTL) candidates.

The model treats promoter single-nucleotide variants (SNVs) associated with the
same tissue and target gene as an unordered candidate set. It combines global
features, within-group relative features, pairwise biological relation biases,
and a prior-residual scoring head to produce a `tpvepr_score` and
`tpvepr_rank` for each candidate.

This user-facing release contains:

- GTEx v11 preprocessing;
- feature construction and chromosome-held-out data splitting;
- the TPVEPR model;
- model training, validation, and testing;
- inference on new tissue-gene candidate sets;
- runnable Python entry points, configuration files, example data, and tests.

The repository intentionally excludes historical training artifacts,
pretrained weights, ablation experiments, and baseline comparisons.

> Scope: TPVEPR performs relative ranking within a tissue-gene candidate set.
> It is not a significant-versus-nonsignificant classifier, and it does not
> interpret association strength as causal probability or clinical
> pathogenicity.

## 1. Project structure

```text
TPVEPR/
├── configs/
│   ├── tpvepr.json              # Full model configuration
│   └── smoke_test.json          # Small end-to-end test configuration
├── examples/
│   └── example_training_input.tsv
├── src/tpvepr/
│   ├── config.py                # Configuration and validation
│   ├── dataset.py               # Grouped datasets and batching
│   ├── features.py              # Coordinates and derived features
│   ├── io.py                    # Table and JSON input/output
│   ├── model.py                 # TPVEPR architecture
│   ├── predict.py               # Checkpoint loading and inference
│   ├── prepare.py               # General training-data preparation
│   ├── preprocessing.py         # GTEx v11 preprocessing
│   └── training.py              # Losses, training, and evaluation
├── tests/
│   └── test_tpvepr.py
├── run_preprocess.py            # Entry point 1: GTEx preprocessing
├── run_prepare_dataset.py       # Entry point 2: dataset preparation
├── run_train.py                 # Entry point 3: training
├── run_predict.py               # Entry point 4: inference
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

## 2. Environment and dependencies

Python 3.11 is recommended. Python 3.10 to 3.12 is supported.

The verified environment is:

| Package | Verified version | Purpose |
|---|---:|---|
| Python | 3.11.5 | Runtime |
| NumPy | 2.4.6 | Numerical operations |
| Polars | 1.42.0 | Large-table processing |
| PyArrow | 24.0.0 | Parquet support |
| PyTorch | 2.12.1 | Model training and inference |
| pytest, optional | 9.0.2 | Automated tests |

### Windows PowerShell

```powershell
cd TPVEPR
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-deps
```

### Linux and macOS

```bash
cd TPVEPR
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-deps
```

For NVIDIA GPU use, install the PyTorch build that matches the local CUDA
environment first. Then install the remaining dependencies and this package:

```bash
pip install numpy==2.4.6 polars==1.42.0 pyarrow==24.0.0
pip install -e . --no-deps
```

The default `"device": "auto"` setting uses CUDA when available and otherwise
uses the CPU. A command can override this setting with `--device cpu`,
`--device cuda`, or `--device cuda:0`.

## 3. Input data

The software accepts Parquet, TSV, TXT, and CSV files. Parquet is recommended
for large datasets.

All genomic coordinates must use GRCh38 and a 1-based inclusive convention. Do
not mix genome builds within one training or prediction run.

### 3.1 Required training columns

| Column | Type | Description |
|---|---|---|
| `tissue` | string | Tissue name |
| `gene_id` | string | Target gene identifier |
| `chr` | string | Variant chromosome, such as `chr1` or `1` |
| `variant_pos` | integer | 1-based genomic coordinate of the SNV |
| `ref` | string | Single reference allele: A, C, G, or T |
| `alt` | string | Single alternative allele: A, C, G, or T |
| `strand` | string | Gene strand: `+` or `-` |
| `gene_start` | integer | 1-based inclusive gene start |
| `gene_end` | integer | 1-based inclusive gene end |
| `pval_nominal` | float | Nominal association P value in `[0, 1]` |

The training-data builder defines:

```text
rank_group_id = tissue + "::" + gene_id
relevance = min(300, -log10(pval_nominal))
```

The capped relevance of 300 is assigned when `pval_nominal=0`. Every
`rank_group_id` must contain at least two candidate variants.

### 3.2 Recommended optional columns

| Column | Type | Default behavior |
|---|---|---|
| `variant_id` | string | Generated as `chr_pos_ref_alt` |
| `gene_chr` | string | Copied from `chr` |
| `biotype` | string | Set to `unknown` |
| `af` | float | Set to 0 |
| `ma_samples` | numeric | Set to 0 |
| `ma_count` | numeric | Set to 0 |
| `rel_to_tss` | integer | Derived from strand-aware coordinates |
| `promoter_start` | integer | Derived from the promoter window |
| `promoter_end` | integer | Derived from the promoter window |
| `split` | string | Rebuilt from chromosome by default |

Missing `af`, `ma_samples`, or `ma_count` values do not prevent execution, but
they remove informative allele-support signals. Supply the measured values for
scientific analyses.

### 3.3 TSS and promoter coordinates

The default promoter window is `[-2000, +500]` relative to the transcription
start site (TSS):

```text
Positive strand: rel_to_tss = variant_pos - gene_start
Negative strand: rel_to_tss = gene_end - variant_pos
```

If `rel_to_tss`, `promoter_start`, or `promoter_end` is absent, the software
derives it automatically. Variants outside the promoter window, indels,
multi-nucleotide variants, equal reference and alternative alleles, and
non-ACGT alleles produce an explicit error.

### 3.4 Prediction input and output

Prediction uses the same candidate columns, except that `pval_nominal` and
`split` are not required. Rows from the same candidate set must share the same
`tissue` and `gene_id`.

The output adds:

| Column | Description |
|---|---|
| `rank_group_id` | Candidate-set key, `tissue::gene_id` |
| `tpvepr_score` | Centered within-group score; larger values have higher priority |
| `tpvepr_rank` | Descending within-group rank; 1 is the highest priority |

A singleton group is accepted, but its rank is necessarily 1 and contains no
within-group comparison.

## 4. Complete workflow

### 4.1 Optional: preprocess raw GTEx v11 files

Expected input layout:

```text
GTEx_Analysis_v11_eQTL/
└── GTEx_Analysis_v11_eQTL/
    ├── Adipose_Subcutaneous.v11.eGenes.txt.gz
    ├── Adipose_Subcutaneous.v11.eQTLs.signif_pairs.parquet
    ├── ...
    ├── Whole_Blood.v11.eGenes.txt.gz
    └── Whole_Blood.v11.eQTLs.signif_pairs.parquet
```

Run:

```powershell
python run_preprocess.py --input-root ..\GTEx_Analysis_v11_eQTL --output-root data\gtex_v11
```

This step:

1. reads the eGenes and significant-pairs files for each tissue;
2. retains `protein_coding` and `lncRNA` genes with `qval <= 0.05`;
3. retains SNVs in the strand-aware `[-2000, +500]` promoter window;
4. writes one combined candidate table.

Outputs:

```text
data/gtex_v11/
├── promoter_significant_pairs.parquet
├── preprocess_config.json
└── preprocess_summary.tsv
```

To change the promoter window or gene biotypes:

```powershell
python run_preprocess.py --input-root ..\GTEx_Analysis_v11_eQTL --output-root data\gtex_v11 --promoter-upstream 2000 --promoter-downstream 500 --qvalue-threshold 0.05 --biotypes protein_coding lncRNA
```

### 4.2 Build the ranking dataset

Using the GTEx preprocessing output:

```powershell
python run_prepare_dataset.py --input data\gtex_v11\promoter_significant_pairs.parquet --output data\ranking\promoter_eqtl_ranking.parquet
```

Using a custom TSV, CSV, or Parquet file that follows Section 3:

```powershell
python run_prepare_dataset.py --input path\to\your_training_candidates.tsv --output data\ranking\promoter_eqtl_ranking.parquet
```

The default chromosome split is:

| Split | Chromosomes |
|---|---|
| Train | chr1 to chr16 |
| Validation | chr17 to chr18 |
| Test | chr19 to chr22, chrX, and chrY |

This protocol keeps each chromosome in one partition and reduces local genomic
leakage. To preserve an existing `split` column containing `train`,
`validation`, and `test`, add `--use-existing-split`.

The command also writes `promoter_eqtl_ranking.summary.json` next to the
dataset. The summary records row counts, candidate-set counts, promoter
coordinates, and split statistics.

### 4.3 Train TPVEPR

Use the full model configuration:

```powershell
python run_train.py --dataset data\ranking\promoter_eqtl_ranking.parquet --config configs\tpvepr.json --output-root outputs\tpvepr
```

Force CPU execution:

```powershell
python run_train.py --dataset data\ranking\promoter_eqtl_ranking.parquet --config configs\tpvepr.json --output-root outputs\tpvepr --device cpu
```

Training produces:

```text
outputs/tpvepr/
├── model/
│   └── tpvepr.pt
├── metrics/
│   ├── feature_state.json
│   ├── ranking_metrics.tsv
│   └── training_history.tsv
├── predictions/
│   └── test_predictions.tsv
└── run_config.json
```

The checkpoint stores model parameters, training-split numerical statistics,
categorical vocabularies, the model configuration, and the promoter window.
Inference must reuse these values rather than refitting preprocessing
statistics.

### 4.4 Rank new candidate variants

```powershell
python run_predict.py --input path\to\new_candidates.tsv --checkpoint outputs\tpvepr\model\tpvepr.pt --output outputs\new_candidate_ranking.tsv
```

GPU example:

```powershell
python run_predict.py --input path\to\new_candidates.tsv --checkpoint outputs\tpvepr\model\tpvepr.pt --output outputs\new_candidate_ranking.tsv --device cuda
```

Categories not observed in the training split, including tissues or
chromosomes, map to `<UNK>`. The vocabulary is not expanded during inference.

## 5. One-minute smoke test

The example table contains one candidate set in each of the training,
validation, and test partitions. It tests the software interface only and has
no scientific interpretation.

```powershell
python run_prepare_dataset.py --input examples\example_training_input.tsv --output data\smoke\ranking.parquet
python run_train.py --dataset data\smoke\ranking.parquet --config configs\smoke_test.json --output-root outputs\smoke
python run_predict.py --input examples\example_training_input.tsv --checkpoint outputs\smoke\model\tpvepr.pt --output outputs\smoke\predictions.tsv --device cpu
```

Run the automated tests:

```powershell
pip install -r requirements-dev.txt
pytest -q
```

## 6. Model and training configuration

`configs/tpvepr.json` defines the released architecture:

- 64-dimensional candidate states;
- four attention heads;
- two relation-aware Transformer blocks;
- a feed-forward width of 128;
- categorical embeddings of width 8;
- dropout of 0.10;
- AdamW with a learning rate of `6e-4` and weight decay of `2e-4`;
- a maximum candidate-set size of 256;
- a maximum padded candidate-pair budget of 120,000;
- a composite listwise, Lambda-style pairwise, tie-aware top-one, and
  calibration loss;
- early stopping on a validation composite score with patience 2.

The relation bias uses TSS distance, relative promoter position, allele
frequency, minor-allele sample count, and minor-allele count. The architecture
uses no positional encoding because candidate order has no biological meaning.

## 7. Resource guidance

- At least 32 GB of memory is recommended for the full GTEx dataset.
- A GPU is optional. CPU training is supported but takes longer.
- If memory is insufficient, reduce `max_pair_tokens` and
  `max_batch_groups`. Do not truncate candidate sets by their labels.
- If a real candidate set exceeds `max_group_size`, increase that setting.
  The software raises an error instead of silently dropping rows.
- Prefer Parquet over TSV for million-row datasets.

## 8. Python API

The installed package can also be called directly:

```python
from pathlib import Path

from tpvepr.config import TPVEPRConfig
from tpvepr.predict import predict
from tpvepr.prepare import prepare_training_dataset
from tpvepr.training import train

prepare_training_dataset(
    Path("candidates.tsv"),
    Path("ranking.parquet"),
)

config = TPVEPRConfig.from_json(Path("configs/tpvepr.json"))
train(
    Path("ranking.parquet"),
    Path("outputs/tpvepr"),
    config,
)

predict(
    input_path=Path("new_candidates.tsv"),
    checkpoint_path=Path("outputs/tpvepr/model/tpvepr.pt"),
    output_path=Path("outputs/new_ranking.tsv"),
)
```

## 9. Interpretation and limitations

- `tpvepr_rank=1` identifies the highest-priority candidate within one
  tissue-gene candidate set. It is not an absolute risk score across genes or
  tissues.
- Training relevance is derived from eQTL association P values. Association
  strength can reflect allele frequency, sample size, linkage disequilibrium,
  and measurement error.
- The current model does not directly use sequence embeddings, transcription
  factor motifs, chromatin accessibility, tissue-specific epigenomic data, or
  fine-mapping probabilities.
- External applications should verify the genome build, tissue definition,
  gene-annotation version, and allele orientation. Findings require validation
  in an independent cohort or functional assay.
