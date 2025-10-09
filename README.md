# ReMedQA

This repository contains the dataset and code for our paper: **"ReMedQA: Are We Done With Medical Multiple-Choice Benchmarks?"**

Medical MCQA benchmarks report near-human accuracy, but accuracy alone is not a reliable measure of competence. Models often change their answers under minor perturbations, revealing a lack of robustness. **ReMedQA** addresses this gap by augmenting standard medical MCQA datasets with **open-answer variants** and **systematically perturbed items**, enabling fine-grained evaluation of model reliability.

We also introduce two new metrics:  
- 🧠 **ReAcc** – measures correctness across all variations  
- 🔁 **ReCon** – measures consistency regardless of correctness

Our findings show that high accuracy can mask low reliability. Large models often exploit structural cues, while smaller models are underestimated by standard MCQA. Despite near-saturated accuracy, we are **not** yet done with medical multiple-choice benchmarks.
