<div align="center">

# PBa-LLM: Privacy- and Bias-aware NLP in General and Applied to AI-based Recruitment

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)

**Gonzalo Mancera · Daniel DeAlcala · Julian Fierrez · Ruben Tolosana · Francisco Jurado · Alvaro Ortigosa · Aythami Morales**

*BiometricsAI, Universidad Autónoma de Madrid (UAM)*

</div>

---

## 📄 Overview

The advancement of Large Language Models (LLMs) has enabled powerful NLP applications, but it also raises serious concerns about personal data exposure and fairness in sensitive domains such as recruitment. **PBa-LLM** is a framework that uses Named Entity Recognition (NER) to anonymize sensitive information—names, locations, and organizations—before training or deploying language models, aiming to protect user privacy while preserving downstream task performance.

We evaluate **eight anonymization models** (specialized NER systems and general-purpose LLMs) across **six text classification datasets**, and apply the framework to a **bias-aware CV screening case study** on FairCVdb, showing that privacy-preserving preprocessing can be integrated into real-world recruitment pipelines without degrading task performance.

<p align="center">
  <img src="assets/fig1_graphical_abstract1.png" width="600">
  <br>
  <em>Fig. 1 — Graphical abstract: models are trained on original text (𝒟) and on NER-anonymized text (𝒟⁻) to compare performance.</em>
</p>

## 🔑 Key Contributions

- A NER-based anonymization pipeline that removes Person, Location, and Organization entities from text prior to LLM training, formalized as a comparability condition between models trained on original and anonymized data.
- A broad, controlled comparison of **eight anonymization systems** — Presidio, Flair, Stanza, NER-CoNLL2003-BERT, GPT-3.5, GPT-4 Mini, GPT-4 Nano, and DeepSeek-V3 — spanning both specialized NER architectures and general-purpose LLMs.
- Evaluation across **six diverse NLP datasets**: Sentiment140, DBPedia, BBC News, IMDB, Cyberbullying Classification, and News Category Dataset.
- A real-world case study on **AI-based blind recruitment** using FairCVdb, integrating anonymization with gender-neutral preprocessing to promote fairer candidate scoring.

## 🏗️ Framework

The core of PBa-LLM is a modular, model-agnostic pipeline in which an anonymization module (N) and a downstream language model (M) are fully interchangeable components. Raw text is first tokenized and passed through the NER anonymizer, which detects and masks Person, Location, and Organization entities, producing an anonymized version of the text. Both the original and anonymized text are then independently used to train a BERT-based classifier, allowing a direct comparison of downstream performance under the two conditions.

<p align="center">
  <img src="assets/fig2_framework.png" width="700">
  <br>
  <em>Fig. 2 — Privacy-aware training pipeline: text is anonymized by the NER module (N) before being passed to the downstream BERT-based classifier (M).</em>
</p>

## 💼 Case Study: Blind Recruitment

To demonstrate the practical value of the framework, PBa-LLM is applied to an AI-based recruitment scenario using the FairCVdb dataset. Candidate resumes are anonymized before being processed by the scoring system, removing personally identifiable information such as names and locations. This is combined with a gender-neutral preprocessing step that removes explicit and implicit gender indicators from free-text biographies, aiming to reduce the model's reliance on demographic proxies and produce more balanced candidate rankings.

<p align="center">
  <img src="assets/fig3_recruitment_case_study.png" width="700">
  <br>
  <em>Fig. 3 — Recruitment case study: candidate resumes are anonymized before occupancy prediction and scoring, reducing reliance on personal and demographic information.</em>
</p>

Aquí tienes el resto del README, continuando justo después de la sección de "Case Study" que ya tienes:

markdown
## 📁 Repository Structure

pba-llm/
├── data/ # Dataset loading & preprocessing scripts
├── anonymization/ # NER-based anonymization pipeline
│ ├── models/ # Presidio, Flair, Stanza, NER-CoNLL2003-BERT wrappers
│ └── prompts/ # Prompt templates for GPT-3.5, GPT-4 Mini/Nano, DeepSeek-V3
├── training/ # BERT classifier training scripts (Section 4 experiments)
├── recruitment_case_study/ # FairCVdb occupancy & scoring experiments (Section 5)
├── configs/ # Hyperparameters, model versions, decoding settings
├── results/ # Output tables and figures
├── assets/ # Figures used in this README
├── requirements.txt
└── README.md


## ⚙️ Installation

```bash
git clone https://github.com/<tu-usuario>/pba-llm.git
cd pba-llm
pip install -r requirements.txt
```

## 🚀 Usage

```bash
# Run anonymization on a given dataset with a chosen NER/LLM model
python anonymization/run_anonymization.py --dataset dbpedia --model flair

# Train the downstream BERT classifier on anonymized/non-anonymized data
python training/train_classifier.py --dataset dbpedia --anonymized true

# Run the FairCVdb recruitment case study
python recruitment_case_study/run_case_study.py --transformer roberta --anonymizer gpt-4
```

## 🔬 Reproducibility

Details on model versions, prompt templates, decoding parameters (temperature, top-p), and training hyperparameters (seeds, optimizer settings, number of runs) used in the paper are documented in [`configs/`](configs/).

## 📚 Citation

This work builds on a preliminary version presented at ICDAR 2025:

```bibtex
@inproceedings{mancera2025pba,
  title={PBa-LLM: Privacy-and bias-aware NLP using named-entity recognition (NER)},
  author={Mancera, Gonzalo and Morales, Aythami and Fierrez, Julian and Tolosana, Ruben and Pe{\~n}a, Alejandro and Lopez-Duran, Miguel and Jurado, Francisco and Ortigosa, Alvaro},
  booktitle={International Conference on Document Analysis and Recognition},
  pages={3--20},
  year={2025},
  organization={Springer}
}
```

## 🙏 Acknowledgments

This study has been supported by the projects M2RAI (PID2024-160053OB-I00, MICIU/FEDER) and Cátedra ENIA UAM-VERIDAS en IA Responsable (NextGenerationEU PRTR TSI-100927-2023-2). The work of G. Mancera is supported by FPI-PRE2022-104499 MICINN/FEDER. This work has been conducted within the ELLIS Unit Madrid.

## 📬 Contact

For questions about this work, please contact **gonzalo.mancera@uam.es** or open an issue in this repository.
