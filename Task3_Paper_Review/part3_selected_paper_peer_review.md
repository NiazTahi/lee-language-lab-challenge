# Part 3: Selected Paper and Scientific Peer Review

## Selected Paper

**Paper:** *MDPO: Conditional Preference Optimization for Multimodal Large Language Models*  
**Authors:** Fei Wang, Wenxuan Zhou, James Y. Huang, Nan Xu, Sheng Zhang, Hoifung Poon, Muhao Chen  
**Version read:** arXiv:2406.11839v2, October 7, 2024  
**Paper URL:** https://arxiv.org/pdf/2406.11839

Note: this paper is currently an arXiv paper rather than an *ACL main-conference paper. I am using it here because it was specifically selected for the presentation.

## Review Summary

The paper studies why Direct Preference Optimization (DPO), which is effective for language-only LLM alignment, can behave inconsistently when applied to multimodal large language models. The authors identify an "unconditional preference" problem: during multimodal DPO, models may learn preference patterns from the text alone and underuse the image condition. They demonstrate this with a controlled comparison where DPO trained without image inputs performs similarly to DPO trained with images on MMHalBench.

To address this, the paper proposes MDPO, a multimodal DPO objective with two additional components. First, conditional preference optimization creates image-contrastive preference pairs, encouraging the model to prefer the response conditioned on the correct image over a visually degraded image. Second, anchored preference optimization regularizes the reward of the chosen response so that preference learning does not reduce the likelihood of good chosen responses. Experiments on Bunny-v1.0-3B and LLaVA-v1.5-7B show improvements over standard DPO on MMHalBench, Object HalBench, and AMBER, especially in reducing hallucination.

## Strengths

The paper identifies a clear and important failure mode in multimodal preference optimization. The controlled "DPO without images" experiment is particularly useful because it challenges the assumption that adding multimodal preference data automatically makes the optimization multimodal. This is a strong diagnostic contribution.

The proposed method is intuitive and well connected to the diagnosis. If the model ignores images, then introducing image-contrastive preference pairs directly targets the issue. The conditional preference objective is simple enough to be practical, and the use of cropped images as hard negatives is easy to understand and implement.

The experimental evaluation is reasonably broad for the paper's scope. The authors test two model sizes, use three hallucination-oriented benchmarks, and include ablations showing that conditional preference optimization is the more important MDPO component. The human evaluation result also supports the automatic metrics: MDPO outputs are judged better or equal to DPO outputs on most compared instances.

The paper is also relevant beyond multimodal hallucination. The broader idea is that preference optimization can ignore parts of the conditioning input. This may matter in other settings where inputs have multiple components, such as retrieval-augmented generation, multilingual prompting, or instruction following with structured context.

## Weaknesses

The main limitation is that the evaluation focuses heavily on hallucination benchmarks. This is appropriate for the stated motivation, but it leaves open whether MDPO improves general multimodal instruction following, visual reasoning, OCR-heavy tasks, chart understanding, or multilingual multimodal tasks.

The construction of rejected images is simple, but it may not always produce semantically meaningful negatives. Cropping 0-20% of the image works well in the reported experiments, but for some images a small crop may remove the crucial object, while for others it may preserve enough information. More analysis of when the negative image construction succeeds or fails would strengthen the paper.

The method adds objectives and training complexity compared with standard DPO. The paper reports strong gains, but it would be helpful to see more detail about compute overhead, sensitivity to hyperparameters such as beta and the anchor value, and robustness across seeds.

Finally, the paper evaluates only two base models. Bunny-v1.0-3B and LLaVA-v1.5-7B are useful choices, but stronger or newer multimodal models may behave differently. The paper would be stronger with evidence across more model families and with open reproduction scripts.

## Questions for the Authors

1. How sensitive is MDPO to the crop ratio used for constructing rejected images?
2. Does MDPO improve non-hallucination tasks such as OCR, chart QA, spatial reasoning, or detailed captioning?
3. What is the extra training cost compared with standard DPO?
4. Does the unconditional preference problem appear in multilingual multimodal settings, where the model may ignore either the image or the non-English language?
5. Can the conditional preference idea be extended to retrieval-augmented generation, where the model might ignore retrieved documents?

## Recommendation

**Recommendation: Weak Accept.**

The paper makes a meaningful contribution by diagnosing a concrete failure mode of multimodal DPO and proposing a targeted, effective fix. The method is simple, empirically useful, and supported by ablations. I choose weak accept rather than strong accept because the evaluation is still concentrated on hallucination benchmarks and two model families, and the negative-image construction deserves deeper analysis. Still, the core idea is valuable and likely to influence future multimodal preference optimization work.

## Suggested Presentation Structure

### Slide 1: Title and Framing

- Paper title, authors, arXiv version
- One-sentence thesis: multimodal DPO may ignore the image condition

### Slide 2: Background

- DPO aligns models using chosen/rejected response pairs
- In multimodal DPO, preferences should depend on both image and text
- But standard DPO can learn language-only preference shortcuts

### Slide 3: Key Diagnosis

- DPO without images performs similarly to DPO with images on MMHalBench
- This suggests an unconditional preference problem
- The model may optimize response preference without grounding in the visual input

### Slide 4: MDPO Method

- Standard multimodal DPO objective
- Conditional preference optimization over image pairs
- Anchored preference optimization for chosen responses

### Slide 5: Results

- MDPO improves Bunny-v1.0-3B and LLaVA-v1.5-7B over DPO
- Gains appear on MMHalBench, Object HalBench, and AMBER
- Strongest impact is hallucination reduction

### Slide 6: Strengths

- Clear diagnosis
- Simple method
- Useful ablations
- Broader implication for conditional preference learning

### Slide 7: Weaknesses

- Evaluation mainly focuses on hallucination
- Negative image construction may be task/image dependent
- Limited model families and compute-overhead discussion

### Slide 8: Recommendation

- Weak Accept
- Strong idea, useful experiments, but needs broader evaluation
