# Source-to-module map

The public code is a structured refactor of the research scripts used for the published T790M study. Machine-specific paths, clinical spreadsheets, medical images, intermediate outputs, and duplicate plotting variants were excluded.

| Original analysis stage | Public module |
|---|---|
| Image/ROI conversion and local entropy (`step1`-`step4`) | `t790m_radiomics/entropy.py` |
| Patient-level 30-superpixel partition (`step5`-`step7`) | `t790m_radiomics/partition.py` |
| Population K selection and subregion remapping (`step8`-`step10`) | `t790m_radiomics/partition.py` |
| PyRadiomics extraction scripts | `t790m_radiomics/radiomics.py`, `configs/radiomics.yaml` |
| ComBat scripts | `t790m_radiomics/harmonize.py` |
| Repeated-segmentation reliability analysis | `t790m_radiomics/reliability.py` |
| Mann-Whitney, LASSO/logistic, validation scripts | `t790m_radiomics/modeling.py` |
| SHAP scripts | `t790m_radiomics/explain.py` |

The refactor keeps the paper's method stages and turns the original spreadsheet-by-spreadsheet workflow into parameterized commands. Re-running the published numerical results still requires the original cohorts, masks, data splits, and approved clinical metadata.
