# Part 2: Volunteer Research Interest Fit Proposal

## Working Title

**Bangla as a Testbed for Efficient Multilingual Adaptation in Low-Resource NLP**

## Motivation and Fit

I am interested in volunteering on multilingual and low-resource NLP, with a special focus on Bangla. As a Bangladeshi speaker, I can contribute both technical work and language-aware judgment to a lab whose research directly aligns with multilinguality, language diversity, low-resource languages, efficient model adaptation, and machine learning for language and culture. Prof. Annie Lee's Lee Language Lab focuses on language diversity and multilingualism, and her official research topics include multilinguality, language diversity, low-resource languages, multicultural bias, and NLP/ML pedagogy.

Bangla is spoken by hundreds of millions of people, yet many modern multilingual systems still underperform on Bangla compared with English and other high-resource languages. The research opportunity is not simply to "add Bangla data," but to ask which adaptation strategies are genuinely effective, efficient, and culturally and linguistically robust for Bangla and related South Asian languages.

## Proposed Volunteer Project

I propose to start with a compact, reproducible research project:

**Research Question:**  
When adapting multilingual models to Bangla, which low-cost strategies provide the best tradeoff among accuracy, compute, data requirements, and robustness?

The initial project would compare:

- Full fine-tuning
- Partial layer freezing
- Adapter-style or LoRA-style parameter-efficient tuning
- Cross-lingual transfer from related languages such as Hindi, Assamese, or other Indic languages
- Model/data selection using typological or proxy-performance signals

The project can begin with classification and sequence labeling tasks, then extend to generation or evaluation tasks once the pipeline is reliable.

## Concrete Tasks

### Task 1: Build a Bangla-Centered Evaluation Matrix

Collect and organize existing Bangla NLP benchmarks and datasets, prioritizing resources that are public and reproducible. Candidate task families:

- Sentiment or social media classification
- Named entity recognition
- Part-of-speech tagging or morphological tagging
- Question answering
- Summarization or controlled generation
- Safety or cultural knowledge evaluation

The output would be a small benchmark table documenting task, source, license, train/dev/test size, label space, evaluation metric, and baseline availability.

### Task 2: Establish Strong Baselines

Run reproducible baselines using:

- Multilingual encoders such as mBERT or XLM-R
- Bangla-specialized encoders such as BanglaBERT
- Bangla generation models such as BanglaT5 where appropriate
- Recent multilingual LLMs under zero-shot or few-shot prompting, where compute permits

The goal is not to maximize leaderboard performance immediately, but to produce clean, reusable baselines with controlled train/test splits, seeds, and documented limitations.

### Task 3: Compare Efficient Adaptation Strategies

Building on the PoS freezing experiment from Part 1, evaluate whether parameter-efficient adaptation gives better compute/performance tradeoffs than full fine-tuning for Bangla.

Candidate comparisons:

- Freeze lower layers versus upper layers
- Freeze embeddings only
- LoRA/adapters versus partial freezing
- Low-data regimes: 50, 100, 500, 1,000 examples
- Cross-lingual transfer from related languages

Metrics:

- Accuracy, macro-F1, or task-specific metrics
- Trainable parameter count
- Runtime
- GPU memory use
- Stability across random seeds

### Task 4: Error Analysis with Bangla-Specific Attention

A useful volunteer contribution would be careful error analysis, not just running models. For Bangla, I would inspect:

- Confusion between named entities and common nouns
- Code-mixed Bangla-English examples
- Dialectal or colloquial forms
- Spelling variation and informal transliteration
- Cultural or location-specific references from Bangladesh
- Script-specific tokenization failures

This is where my Bangladeshi background can add value beyond generic multilingual modeling.

## Expected Contributions

The first volunteer-stage contribution could be:

1. A reproducible Bangla low-resource adaptation benchmark with scripts and documentation.
2. A compute-aware comparison of full fine-tuning, freezing, and parameter-efficient adaptation.
3. A qualitative error analysis identifying where multilingual systems fail on Bangla.
4. A short internal report that could grow into a workshop paper, demo paper, or seed project for future lab work.

## Potential Research Gap

Many multilingual NLP papers report average gains across languages, but low-resource language performance is often uneven. A method that improves the average may still fail on Bangla, code-mixed text, culturally grounded questions, or dialectal usage. My proposed work would focus less on broad claims and more on controlled, language-specific evidence.

This connects naturally to recent work on compact typological representations and proxy-based performance prediction from Prof. Lee's group. A possible extension is to test whether typological distance or proxy-model signals can predict which source languages or model families transfer best to Bangla.

## Possible Side Projects

### Side Project 1: Bangla Error Taxonomy for Multilingual Models

Create a small taxonomy of Bangla-specific model failures, including tokenization errors, named-entity errors, code-mixing errors, cultural-knowledge errors, and formal/informal register mismatches.

### Side Project 2: Bangla Data Quality Audit

Audit publicly available Bangla datasets for duplicates, label noise, machine-translated artifacts, licensing ambiguity, and train/test leakage. This could support future benchmarking work in the lab.

### Side Project 3: Bangla Extension for a Low-Resource Learning Tool

Inspired by low-resource language learning applications, explore a Bangla module for flashcard generation, pronunciation support, or vocabulary learning, with careful attention to cultural examples and dialect/register variation.

## Why This Lab

This lab is a strong fit because the work is not only about model performance; it is about language inclusion, multilingual evaluation, and low-resource adaptation. Prof. Lee's group has recent work on typological language representations, proxy performance prediction, multilingual NLP education, and low-resource language tools. A Bangla-centered project would align with those themes while adding a concrete South Asian language direction.

## Initial Timeline for Volunteering

**Weeks 1-2:**  
Read 6-8 core papers, reproduce the Part 1 experiment, and finalize the Bangla benchmark/task selection.

**Weeks 3-4:**  
Run baseline models on one classification task and one sequence-labeling task.

**Weeks 5-6:**  
Run efficient adaptation experiments: freezing, LoRA/adapters if feasible, and low-data curves.

**Weeks 7-8:**  
Perform error analysis and write a short internal report with tables, limitations, and next-step recommendations.

## Selected References

1. Annie En-Shiun Lee. Official Ontario Tech profile. Research topics include multilinguality, language diversity, low-resource languages, multicultural bias, and efficient model adaptation. https://science.ontariotechu.ca/computer-science/people/undergraduate-faculty/annie-lee.php
2. York Hay Ng, Phuong Hanh Hoang, and En-Shiun Annie Lee. 2025. *Less is More: The Effectiveness of Compact Typological Language Representations*. EMNLP 2025. https://aclanthology.org/2025.emnlp-main.1310/
3. David Anugraha, Genta Indra Winata, Chenyue Li, Patrick Amadeus Irawan, and En-Shiun Annie Lee. 2025. *ProxyLM: Predicting Language Model Performance on Multilingual Tasks via Proxy Models*. Findings of NAACL 2025. https://aclanthology.org/2025.findings-naacl.106/
4. Tai Zhang et al. 2025. *Learning Low-Resource Languages Through NLP-Driven Flashcards: A Case Study of Hokkien in Language Learning Applications*. NAACL 2025 System Demonstrations. https://aclanthology.org/2025.naacl-demo.26/
5. Kosei Uemura, Mason Shipton, and Annie Lee. 2024. *Empowering the Future with Multilinguality and Language Diversity*. Teaching NLP 2024. https://aclanthology.org/2024.teachingnlp-1.10/
6. Abhik Bhattacharjee et al. 2022. *BanglaBERT: Language Model Pretraining and Benchmarks for Low-Resource Language Understanding Evaluation in Bangla*. Findings of NAACL 2022. https://aclanthology.org/2022.findings-naacl.98/
7. Wasi Uddin Ahmad et al. 2023. *BanglaNLG: Benchmarks and Resources for Evaluating Low-Resource Natural Language Generation in Bangla*. Findings of EACL 2023. https://aclanthology.org/2023.findings-eacl.54/
8. Tamzeed Mahfuz et al. 2025. *Too Late to Train, Too Early To Use? A Study on Necessity and Viability of Low-Resource Bengali LLMs*. COLING 2025. https://aclanthology.org/2025.coling-main.79/
9. Md. Abu Sayed et al. 2026. *BanSuite: A Unified Toolkit and Software Platform for Low-Resource NLP in Bangla*. EACL 2026 System Demonstrations. https://aclanthology.org/2026.eacl-demo.44/
10. Sebastian Ruder et al. 2021. *XTREME-R: Towards More Challenging and Nuanced Multilingual Evaluation*. EMNLP 2021. https://aclanthology.org/2021.emnlp-main.802/
11. Pratik Joshi et al. 2020. *The State and Fate of Linguistic Diversity and Inclusion in the NLP World*. ACL 2020. https://aclanthology.org/2020.acl-main.560/
12. Ayomide Odumakinde et al. 2025. *Multilingual Arbitration: Optimizing Data Pools to Accelerate Multilingual Progress*. ACL 2025. https://aclanthology.org/2025.acl-long.939/
