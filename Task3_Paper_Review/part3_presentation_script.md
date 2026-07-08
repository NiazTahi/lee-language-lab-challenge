# Five-Minute Presentation Script

## Slide 1

Today I am reviewing *MDPO: Conditional Preference Optimization for Multimodal Large Language Models* by Wang and colleagues. The main idea is that standard DPO, when applied to multimodal models, may not actually use the image condition strongly enough. The paper proposes MDPO to make preference optimization more visually grounded.

## Slide 2

Direct Preference Optimization, or DPO, aligns a model using chosen and rejected responses. In a language-only setting, the input is usually a text prompt. In a multimodal setting, the preference should depend on both the image and the text question. The problem is that the model may learn shortcuts from the text alone.

## Slide 3

The paper's key diagnostic is very interesting. The authors train a DPO variant where the images are removed from the preference data. Surprisingly, this image-free version performs similarly to normal DPO on MMHalBench. This suggests that standard multimodal DPO can learn unconditional preferences: it optimizes which response sounds better without fully grounding that preference in the image.

## Slide 4

To fix this, the authors propose MDPO. It has two additions to standard multimodal DPO. First, conditional preference optimization contrasts the correct image with a degraded image, while keeping the question and chosen response fixed. This forces the model to care about visual information. Second, anchored preference optimization regularizes the reward of the chosen response so that training does not reduce the likelihood of good responses.

## Slide 5

The experiments use Bunny-v1.0-3B and LLaVA-v1.5-7B, with evaluation on MMHalBench, Object HalBench, and AMBER. MDPO consistently improves over standard DPO. For Bunny, the MMHalBench score increases from 2.28 with DPO to 2.96 with MDPO, and the hallucination rate drops from 0.56 to 0.42. For LLaVA-v1.5-7B, MDPO also improves hallucination-related metrics.

## Slide 6

The paper has several strengths. The diagnosis is clear and convincing, especially the no-image DPO experiment. The method follows naturally from the diagnosis: if the model ignores images, add an objective where the image is the contrastive variable. The ablations also help, showing that conditional preference optimization is the most important component.

## Slide 7

My main concerns are about scope and robustness. The evaluation focuses mostly on hallucination benchmarks, so it is less clear whether MDPO improves OCR, chart understanding, spatial reasoning, or general instruction following. The rejected-image construction is also simple: cropping works well here, but it may behave differently depending on the image and task. I would also like more reporting on compute overhead and sensitivity across random seeds.

## Slide 8

My recommendation is weak accept. The paper identifies an important failure mode in multimodal preference optimization and proposes a simple, effective correction. I would not call it a complete solution yet, because the evaluation should be broader, but the core idea is strong and likely useful for future multimodal alignment work.
