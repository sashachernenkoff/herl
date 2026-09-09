# <img src="assets/herl_logo.png" alt="PASTL logo" height="100"><br>HERL: Heritability Estimation by Representation Learning

Deep generative modelling and representation learning of high-dimensional 
genotype data for heritability estimation using a categorical variational 
autoencoder (VAE).

![Python](https://img.shields.io/badge/python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)

## Background

Standard genomic-heritability methods (e.g. GCTA-GREML) build a genetic 
relationship matrix (GRM) directly from standardized SNP genotypes, which 
is a strictly linear summary of relatedness. This project explores whether 
a VAE that compresses each individual's genotype into a low-dimensional 
representation can provide input for a GRM that yields improved heritability 
estimates. The latent representations should be able to capture nonlinear 
structure (dominance, epistasis, population structure) that a linear GRM 
cannot.

![Method overview](assets/fig1.png)

## Method overview

The pipeline has four core stages:

```
raw PLINK binaries (.bed/.bim/.fam)
        │
        ▼
  ┌───────────────┐   Automated QC + optional LD pruning via PLINK
  │ 1. Preprocess │   PyArrow ingest → imputed NumPy binary (.npy)
  └───────────────┘
        │  genotype dosages (individuals × SNPs)
        ▼
  ┌───────────────┐   Categorical VAE encoder → latent code z per individual
  │ 2. Represent  │   (fixed latent dim compresses the SNP markers)
  └───────────────┘
        │  Y = matrix of latent means  (n individuals × d latent dims)
        ▼
  ┌───────────────┐   GRM = (1/d) · Z Zᵀ        (Z = standardized latent, d dims)
  │ 3. Relate     │   written in GCTA binary format
  └───────────────┘
        │  GRM + phenotype (case/control, or gene expression)
        ▼
  ┌───────────────┐   GCTA --reml  →  V(G)/Vp   (SNP heritability, h²)
  │ 4. Estimate   │
  └───────────────┘
```

### 1. Preprocessing & LD pruning

HERL operates directly on raw PLINK binaries. The preprocessing modules drive 
a PLINK pipeline before neural-network training:
1. **Quality Control:** Filters on missingness, low Minor Allele Frequency 
(MAF), and Hardy-Weinberg violations.
2. **LD Pruning (optional):** Iteratively prunes correlated markers 
(`--indep-pairwise`).
3. **Ingestion:** Recodes the retained markers to `.raw`, parses them into 
memory via PyArrow, mean-imputes residual missingness, and caches a NumPy 
binary (`.npy`, tagged pruned/unpruned) for fast subsequent loads.

### 2. Representation learning (Categorical VAE)

A fully connected VAE encodes the imputed genotype tensor into a latent 
distribution and reconstructs it. The encoder and decoder are symmetric 
fully-connected stacks - `SNPs → 512 → 256 → 128` (latent) and 
`128 → 256 → 512 → SNPs` - with batch normalization and ReLU activations.
- **Continuous dosage input:** The encoder consumes the mean-imputed allele 
dosages directly (one value per SNP) and maps them to the Gaussian posterior 
parameters (`mu`, `log_var`).
- **Categorical reconstruction (dominance effects):** The decoder outputs a 
3D tensor of logits (`Batch, SNPs, 3`) - a distribution over the three genotype 
states for every SNP. Modelling each genotype categorically, rather than as a 
single additive dosage, lets the decoder represent non-additive (dominance) 
effects, i.e. departures from Hardy-Weinberg proportions.
- **Objective:** The VAE minimizes the Evidence Lower Bound (ELBO): a 
categorical cross-entropy reconstruction term plus the KL divergence between 
the posterior and a standard-normal prior.
- **Training dynamics:** Adam with L2 weight decay, mixed-precision (bf16) 
autocasting, gradient accumulation, and gradient-norm clipping. Both the 
best-validation-loss and the final-epoch weights are checkpointed, and metrics 
are logged to Weights & Biases.

### 3. Genetic relationship matrix (GRM)

The per-individual latent means (`mu`) form a representation matrix `Y` 
(individuals × latent dimensions). Each latent dimension (column) is 
standardized to mean 0 and unit variance to give `Z`, and the GRM is computed 
as `GRM = (1/d) · Z Zᵀ`, where `d` is the number of latent dimensions. This 
yields a positive semi-definite matrix with diagonal ≈ 1. It is then 
serialized to GCTA's binary triplet (`.grm.bin`, `.grm.N.bin`, `.grm.id`) 
alongside a phenotype (`.phen`) file.

### 4. Heritability estimation

Heritability is estimated by restricted maximum likelihood (GREML) with 
[GCTA](https://yanglab.westlake.edu.cn/software/gcta/) (`--reml`), reporting 
`V(G)/Vp`. [LDAK](https://dougspeed.com/ldak/) is supported as an alternative.

## Reconstruction quality & determinism

While the network optimizes a categorical cross-entropy likelihood, 
quantitative-genetics tools (like GRMs) require a single continuous dosage per 
genotype to compute covariance.

To turn the VAE output into a deterministic dosage, the evaluation loop 
marginalizes the predicted probabilities into an expected dosage: 
`Expected Dosage = P(heterozygous) + 2 · P(homozygous alt)`. This preserves the 
soft probabilistic uncertainty the VAE learned without breaking downstream 
mathematical compatibility.

Training is monitored with reconstruction R² (variance explained by the 
expected dosage), the cross-entropy reconstruction error, and the separate 
reconstruction / KL loss components.

## Results

TBD

## Repository layout

The code is organized by pipeline stage:

```
preprocessing/  Preprocess — automated PLINK QC + PyArrow data loading
  wtccc.py                 QC + optional LD pruning + imputation for WTCCC genotypes
  gtex.py                  per-gene cis genotype windows (GTEx)
vae/            Represent  — configurable VAE library + training scripts
  encoder.py               Encoder module (dosages → Gaussian posterior)
  decoder.py               Categorical Decoder module (latent → per-SNP logits)
  loss.py                  ELBO calculation (cross-entropy + KL divergence)
  eval.py                  Metrics (R-squared, reconstruction error)
  model.py                 VAE architecture definition (nn.Module)
  configs/                 JSON configs: wtccc, gtex_cis
  train_wtccc.py           Execution script: coordinates training loop + wandb
  train_gtex_cis.py        Execution script for GTEx gene cis window
heritability/   Relate & Estimate — GRM construction, GCTA runner
  grm_wtccc.py             build GRM + phenotype + GCTA binaries (WTCCC)
  grm_gtex_cis.py          build GRM + GCTA binaries (GTEx cis)
  gcta.py                  run GCTA GREML in parallel across directories
  aggregate.py             crawl .hsq outputs → single results CSV
eda/            Analyze    — exploratory data analysis
  analyze_sparsity.py      MAF / missingness / genotype spectra + PCA variance ceiling
  analyze_ld_structure.py  LD decay and haplotype block-size distributions
```

Datasets covered: **WTCCC** (seven case/control diseases: BD, CAD, CD, HT, RA, 
T1D, T2D) and **GTEx** (gene-level / cis-eQTL).


## Usage

The stages are separate command-line scripts that share `--data_path` / 
`--output_dir` arguments. A typical WTCCC run for one disease:

```bash
# 1. Train the VAE and write latent representations + expected reconstructions
#    [Optional: append --pruned to use LD-pruned SNPs]
python vae/train_wtccc.py \
    --config vae/configs/wtccc.json --disease BD \
    --data_path /path/to/wtccc/BD \
    --output_dir /path/to/output/WTCCC/BD/BD_128

# 2. Build the GRM (from the latents) + phenotype + GCTA binaries
python heritability/grm_wtccc.py \
    --disease BD \
    --output_dir out/WTCCC/BD/BD_128

# 3. Estimate heritability with GCTA (external tool)
gcta64 --reml \
    --grm-bin out/WTCCC/BD/BD_128/BD \
    --pheno   out/WTCCC/BD/BD_128/BD.phen \
    --out     out/WTCCC/BD/BD_128/BD_reml
```

Each training run writes, into `--output_dir`, the latent means 
(`{disease}_latent_representations.csv`, with a leading `IID` column so downstream 
GRM/phenotype construction joins individuals by ID), the expected-dosage reconstructions 
(`{disease}_reconstructed_data.csv`), and the model weights - both the 
final-epoch (`{disease}_vae.pt`) and best-validation-loss (`{disease}_vae_best.pt`) 
checkpoints.

### Input format

Genotype input relies on prefix paths to raw PLINK binary triplets (`.bed`, 
`.bim`, `.fam`). WTCCC case/control status is inferred from the individual ID 
prefix in the `.fam` file.


## Requirements

- Python 3.9+
- `torch`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `pyarrow`, `wandb`
- External: [PLINK 1.9](https://www.cog-genomics.org/plink/) and [GCTA](https://yanglab.westlake.edu.cn/software/gcta/)

The models are implemented in **PyTorch** and run on GPU when available (CPU otherwise).
