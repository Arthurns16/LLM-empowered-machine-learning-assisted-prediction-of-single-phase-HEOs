# Rational Design of Single-Phase High Entropy Oxides via Large Language Model Data Mining and Explainable Machine Learning

## Summary
This study leverages the large language model **GPT-OSS-120B** to automatically extract high-entropy oxide (HEO) data from scientific literature with **96% accuracy**, enabling the construction of high-quality machine-learning datasets.
On top of the base LLM-generated dataset, **new dataset variants** were created by incorporating **advanced statistical descriptors** and **property–synthesis interaction features**, followed by systematic **feature selection** to refine predictive inputs.
Multiple ML models were trained on these diverse datasets to classify HEO crystal structures. The best-performing **multiclass model (XGBoost)** achieved an **F1-score of ~86%** using a feature set that combined primary descriptors with advanced statistical features.
Building on this optimal dataset, a **binary neural-network classifier** reached **97.9% accuracy** in distinguishing **perovskite vs. non-perovskite** compositions.
Interpretability analysis using **SHAP** revealed physically meaningful patterns, demonstrating that the methodology not only delivers strong predictive performance but also provides insights into the mechanisms of HEO phase formation.

## Repository Contents
This repository is organized to support reproducibility while keeping the structure concise:
- **Benchmark datasets** used to evaluate GPT-OSS-120B agents.
- **Primary and featurized datasets**, including engineered and feature-selected variants used for modeling.
- **Automated pipelines** for feature engineering, feature selection, and batch model training with cross-validated hyperparameter tuning.
- **Trained machine-learning models**, including the best-performing multiclass XGBoost classifier and a binary neural-network model.
- **Evaluation artifacts**, such as test-set metrics, classification reports, and dataset/model metadata.

## Citation

@ARTICLE{Da_Silva_Sousa_Santos2026-cn,
  title     = "Rational design of single-phase high-entropy oxides via large
               language model data mining and explainable machine learning",
  author    = "da Silva Sousa Santos, Arthur and Stojanovska, Elena and Alves,
               Jr, Antonio Augusto and de Paula, Amauri Jardim and de Florio,
               Daniel Zanetti and de Almeida, James Moraes",
  journal   = "J. Chem. Inf. Model.",
  publisher = "American Chemical Society (ACS)",
  volume    =  66,
  number    =  9,
  pages     = "5234--5248",
  month     =  may,
  year      =  2026,
  copyright = "https://creativecommons.org/licenses/by/4.0/",
  language  = "en"
}


## Contact

E-mail: santos.arthur@aluno.ufabc.edu.br


