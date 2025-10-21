# Preregistration: Weight of Reasoning Metrics

## Research Question

Do internal reasoning effort metrics (Activation Energy and Attention Process Entropy) show systematic differences between reasoning tasks and length-matched control tasks in small language models?

## Hypotheses

**Primary Hypothesis**: Activation Energy (AE), when z-scored against length-matched controls, will be significantly higher for reasoning prompts compared to control prompts.

**Secondary Hypothesis**: Attention Process Entropy (APE) will show different patterns between reasoning and control tasks, though the direction of effect is exploratory.

## Primary Endpoint

**Cohen's d > 0.5** for AE separation (reasoning vs. control) on the tiny evaluation set (n=10 total samples).

## Experimental Design

### Model and Configuration
- **Model**: EleutherAI/pythia-70m-deduped (CPU-friendly)
- **Decoding**: Deterministic (temperature=0.0, top_p=1.0)
- **Max tokens**: 64 new tokens per prompt
- **Seed**: Fixed at 1337 for reproducibility

### Dataset
- **Reasoning tasks**: 5 math/logic word problems requiring step-by-step reasoning
- **Control tasks**: 5 neutral descriptive text prompts of similar length
- **Length matching**: Control prompts designed to generate similar token counts (±10%)

### Metrics
1. **Activation Energy (AE)**: Mean L2 norm of hidden states in reasoning window, normalized by token count
2. **Attention Process Entropy (APE)**: Mean entropy of attention patterns across heads in reasoning window

### Reasoning Window Definition
- Last N generated tokens excluding final answer token(s)
- N = min(32, max_new_tokens - 1) = 31 tokens
- Heuristic approach for this initial validation

## Analysis Plan

### Statistical Tests
- **Primary**: Cohen's d for AE (reasoning vs. control)
- **Secondary**: Cohen's d for APE (reasoning vs. control)
- **Effect size interpretation**: d > 0.5 considered "medium" effect

### Exclusion Criteria
- Generation failures (empty or malformed output)
- Generated text < 3 tokens
- NaN values in metric computation (due to model errors)

### Success Criteria
- **AT-1**: Environment setup completes without errors
- **AT-2**: Tiny evaluation completes in <5 minutes on CPU
- **AT-3**: All unit tests pass
- **Primary**: Cohen's d(AE) > 0.5 with n ≥ 8 valid samples per group

## Limitations and Caveats

1. **Small sample size**: n=10 total limits statistical power
2. **Heuristic reasoning window**: May not perfectly separate reasoning from answers
3. **Single model**: Results may not generalize to other architectures
4. **CPU-only**: May miss GPU-specific behaviors
5. **Tiny model**: Pythia-70M may not exhibit strong reasoning patterns

## Future Extensions

This preregistration covers the initial validation. Future work will:
- Scale to larger models and datasets
- Implement more sophisticated reasoning window detection
- Add additional metrics (APL, CUD, SIB, FL)
- Include length-stratified matching and partial correlations

## Preregistration Date

**Date**: 2024-12-19
**Version**: 1.0
**Status**: Initial validation study

## Data and Code Availability

- All code will be made available under open source license
- Raw activations cached for reproducibility
- Evaluation results and plots will be publicly available
- No human subjects data involved
