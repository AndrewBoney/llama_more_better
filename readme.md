this project is designed to try and experiment with improving an llms output by generating multiple answers (hence more / better). this is inspired by [an answer.ai post by jonathan whitaker](https://www.answer.ai/posts/2024-05-17-more-better.html).

I want to flesh this out a bit more with:
- scoring without the use of an extra heavyweight llm.  
- evaluation against common benchmarks

I'm going to start with llama 3.2 1B, either [base](https://huggingface.co/meta-llama/Llama-3.2-1B) or [instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)

# scoring thoughts

my best idea here so far is to have dual encoders of prompt and response, and a mechanism that takes prompts and response embeddings (i.e logits of last token) as inputs and ranks. this could involve a linear projection (initially, could be enhanced) on top of prompts / responses, then train with contrastive learning i.e. similarity of projections against responses labelled 0 / 1. 
dataset could come from an human preference dataset e.g. [trl-lib/zen](https://huggingface.co/datasets/trl-lib/zen) or [ms_marco](https://huggingface.co/datasets/microsoft/ms_marco)

# evaluation
either use huggingface [lm-eval](https://github.com/huggingface/lm-evaluation-harness) or [llama 3.2 evals](https://huggingface.co/collections/meta-llama/llama-32-evals-66f44b3d2df1c7b136d821f0)
