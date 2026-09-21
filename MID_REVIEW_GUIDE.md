# Mid-review walkthrough

1. Show **Real dataset & test** and the official UCI source, population, fields and license.
2. Show the real-data train/validation/test split. Explain that IDs are excluded and duplicates removed before splitting.
3. Show **Borrower profiles**: age, sex, education, marital status and financial history.
4. Generate a fresh batch and download the CSV. Open it in Excel to demonstrate the actual output.
5. Compare fidelity and genuine TSTR against the real-trained baseline.
6. Explain the limitations: Taiwan credit-card benchmark, one seed, no privacy guarantee, no claim of rural Indian validation.

## Packaged final test results

| Method | PR-AUC | F1 |
|---|---:|---:|
| Real-trained reference | 0.5507 | 0.5335 |
| Independent | 0.2554 | 0.3228 |
| Gaussian copula | 0.4911 | 0.4711 |
| VAE only | 0.3558 | 0.4096 |
| Latent diffusion | 0.4296 | 0.4344 |

The Gaussian copula has higher TSTR utility than latent diffusion in this run. Latent diffusion improves over VAE-only, but does not match the real-trained classifier. This is an experimental finding, not a failure to hide.

## Questions

**Is the source real?** Yes: the included original UCI workbook and SHA-256 manifest document it.

**Are generated profiles real people?** No. New model samples have new synthetic IDs, but memorization/privacy still requires evaluation.

**Why no names?** They are absent from the source and irrelevant to demonstrating statistical credit-data synthesis.

**Why not Indian rural borrowers?** This dataset supports a verifiable prototype. Deployment to that population requires relevant real data and external validation.

**Novelty?** TabSyn-inspired architecture and a real-data fidelity/utility evaluation; no claim of inventing latent diffusion.