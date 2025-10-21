# Fine-tuned Model Error Analysis

## Executive Summary

Fine-tuning TinyLlama-1.1B on Alpaca successfully created an instruction-following assistant for in-distribution tasks (**+27.5% METEOR**) but catastrophically failed on GPT-4 quality prompts (**-92.8% BLEU, -33.4% METEOR on OpenOrca**).

**Conclusion**: Model is production-ready for Alpaca-style tasks only. Not suitable for complex reasoning or varied instruction formats.

---

## Performance Overview

| Dataset | Type | Samples | METEOR Δ | ROUGE-L Δ | BLEU Δ | Status |
|---------|------|---------|----------|-----------|--------|--------|
| **Alpaca** | In-dist | 198 | **+27.5%** | **+5.1%** | -2.1% | Success |
| **Dolly-15k** | OOD | 156 | **+26.4%** | **+0.6%** | -19.4% | Good |
| **OpenOrca** | OOD (GPT-4) | 111 | **-33.4%** | **-17.3%** | **-92.8%** | Critical Failure |

**Key Insight**: METEOR improvements show semantic learning, but catastrophic BLEU collapse on OpenOrca indicates structural/reasoning failure.

---

## Error Pattern Categories

### 1. Complex Reasoning Loss (Model Scale Limitation)

**Problem**: Model fails multi-step logic and factual accuracy on knowledge-intensive tasks.

**Severity**: **Critical** - Causes incorrect/hallucinated answers

#### Example A: Biological Process Reasoning

| Component | Content |
|-----------|---------|
| **Task** | What is the missing first step: "Air goes down esophagus → moves into lungs → CO₂ removed..." |
| **Reference** | "Inhale air through the nose or mouth" |
| **Fine-tuned Output** | "The oxygen to be removed from the air before it reaches the lungs, which is done by taking a deep breath and holding it" |
| **Error Type** | Factual hallucination + incorrect biological sequence |
| **Root Cause** | 1.1B parameters insufficient to retain complex knowledge while learning instruction-following |

#### Example B: Factual Knowledge (A Song of Ice and Fire)

| Component | Content |
|-----------|---------|
| **Task** | "Who were the children of Garth Greenhand?" |
| **Reference** | "Garth the Gardener, John the Oak, Gilbert of the Vines, Brandon of the Bloody Blade..." |
| **Fine-tuned Output** | "Tyrion Lannister, Daenerys Targaryen, Arya Stark, Cersei Lannister, Bran Stark..." |
| **Error Type** | Complete hallucination - confuses unrelated characters |
| **Root Cause** | Model conflates character relationships within same universe |

#### Example C: Historical Facts

| Component | Content |
|-----------|---------|
| **Task** | "Who was the first Time Magazine Man of the Year?" |
| **Reference** | "Charles Lindbergh (1927)" |
| **Fine-tuned Output** | "Albert Einstein in 1927" |
| **Error Type** | Wrong person, plausible year |
| **Root Cause** | Insufficient capacity to maintain factual precision |

#### Example D: Mathematical Reasoning

| Component | Content |
|-----------|---------|
| **Task** | "Calculate atomic mass for lithium" |
| **Reference** | "6.941 u (3 protons + 4 neutrons × 1.007 u each)" |
| **Fine-tuned Output** | "6.941 u. Formula: A = (A + B) × C / D... = 15.37 u" |
| **Error Type** | Nonsensical formula, wrong final answer |
| **Root Cause** | Loss of mathematical reasoning during fine-tuning |

---

### 2. Template Blindness (Structural Mismatch)

**Problem**: Model ignores system prompts and structural constraints present in OpenOrca but absent in Alpaca training.

**Severity**: **Critical** - Fails to follow explicit instructions

#### Example E: System Prompt Ignore

| Component | Content |
|-----------|---------|
| **System Prompt** | "You are a helpful assistant, who always provide explanation. Think like you are answering to a five year old." |
| **Task** | Logic question about premise/hypothesis relationship |
| **Reference** | "Yes, little buddy! If the premise is true... The hypothesis is just another way of saying the same thing..." |
| **Fine-tuned Output** | "Yes, it is possible to conclude that if the premise is true, then so is the hypothesis. In other words, if the conclusion and hypothesis are both based on the same factual statement..." |
| **Error Type** | Ignores "five year old" style constraint entirely |
| **Root Cause** | Training on Alpaca (no system prompts) → cannot handle conditional instructions |

#### Example F: Format Compliance Failure

| Component | Content |
|-----------|---------|
| **Task** | "Classify: 'All lives matter!! I shout it from the roof top!' → Output: 'Hope Speech' or 'Not Hope Speech'" |
| **Reference** | "Hope Speech" |
| **Fine-tuned Output** | "This comment is classified as Not Hope Speech because it does not talk about equality, diversity or inclusion. It is focused on emphasizing the loss of life..." |
| **Error Type** | Adds explanation instead of single-word classification |
| **Root Cause** | Overfitted to conversational format; cannot do structured output |

---

### 3. Contextual Simplification (Overfitting)

**Problem**: Model replaces technical/factual content with generic conversational text.

**Severity**: **High** - Reduces information density and utility

#### Example G: Census Article Task

| Component | Content |
|-----------|---------|
| **Task** | "Write article: How many in percent from census weren't English?" |
| **Reference** | Detailed article with: census methodology explanation, step-by-step calculation (200÷1,000 = 0.2 → 20%), educational narrative |
| **Fine-tuned Output** | "According to 2016 Census, approximately 7.8% of population in Canada is not English speaking... These numbers have been steadily increasing..." |
| **Error Type** | Generic facts, no calculation methodology, wrong context (Canada vs hypothetical) |
| **Root Cause** | Defaults to easy conversational generation instead of structured analysis |

#### Example H: Date Extraction Task

| Component | Content |
|-----------|---------|
| **Task** | Extract dates from: "YouTube CEO is Neal Mohan, appointed February 16, 2023" |
| **Reference** | "February 16, 2023" |
| **Fine-tuned Output** | [Translates entire paragraph to Spanish, then extracts date] |
| **Error Type** | Adds unnecessary translation step |
| **Root Cause** | Cannot handle simple extraction; overgenerates |

---

### 4. Overgeneration Problem

**Problem**: Responses significantly longer than necessary, often incomplete.

**Severity**: **High** - Wastes tokens, increases latency, poor UX

#### Length Analysis

| Dataset | Reference Avg | Fine-tuned Avg | Overgeneration |
|---------|---------------|----------------|----------------|
| Alpaca | 38.7 words | **76.5 words** | **+98%** |
| Dolly-15k | 65.1 words | 72.3 words | +11% |

#### Example I: Arithmetic Exercise

| Component | Content |
|-----------|---------|
| **Task** | "Solve: -236×q + 1943 = -181 for q" |
| **Reference** | Step-by-step solution, final answer: q = 9 |
| **Fine-tuned Output** | "Given that 236 × -q + 1943 = -181, we need to find q such that -q + 1943 = -181. We can solve this equation by finding the value of q such that it satisfies both equations. This gives us q = -30" [INCORRECT + verbose] |
| **Error Type** | Wrong answer + unnecessary verbosity |

#### Example J: Imperative Mood Rewrite

| Component | Content |
|-----------|---------|
| **Task** | Rewrite "Please pick up the book from the shelf" in imperative mood |
| **Reference** | "Pick up the book from the shelf." (7 words) |
| **Fine-tuned Output** | "Please pick up the book from the shelf. The imperative mood is used to request a specific action that must be taken. In this case, 'pick up' means to retrieve something from a location. The imperative mood is used to tell someone to act in a certain way or to indicate a command. It is used to give orders and direct people's behavior. In this context, it is important to use the imperative mood because the person asking for the book has not yet done so themselves. They are simply" [CUTS OFF] (90 words) |
| **Error Type** | 13x longer, doesn't complete, explains instead of answering |

---

## Root Cause Analysis

### Why OpenOrca Fails Catastrophically

1. **Model Scale Bottleneck**
   - TinyLlama 1.1B parameters fundamentally insufficient
   - Cannot retain complex reasoning while learning instruction-following
   - Trade-off: Conversational fluency ↔ Factual accuracy

2. **Training Distribution Mismatch**
   - Alpaca: Simple instructions, no system prompts, conversational style
   - OpenOrca: Complex multi-step tasks, explicit system prompts, structured outputs
   - Model never saw system prompts during training → blind to them

3. **Catastrophic Forgetting**
   - Fine-tuning optimized for Alpaca-style responses
   - Lost base model's technical/factual capabilities
   - LoRA rank (r=16) too low to preserve both capabilities

---

## Recommendations

### Block Deployment

**1. Data Mixing Strategy**
- Add 5-10% OpenOrca examples to training set
- Include system prompt variations
- Force model to learn structural diversity
- **Expected Impact**: +15-20% OpenOrca performance

**2. Model Scale Upgrade**
- Current: TinyLlama 1.1B (bottleneck identified)
- Recommended: Llama-3 8B or Phi-3 Mini (7B)
- **Rationale**: 1.1B cannot handle complex reasoning + instruction-following simultaneously

### High Priority

**3. Increase LoRA Rank**
- Current: r=16
- Recommended: r=32
- **Expected Impact**: Better preservation of base model capabilities, +5-10% parameter efficiency

**4. Length Penalties**
- Implement max_length constraints (target: 1.5x reference length)
- Add early stopping on sentence boundaries
- **Expected Impact**: -40% response length, better completion rate

**5. Structured Output Training**
- Add classification tasks (single-word outputs)
- Include extraction tasks (dates, entities)
- Train on format-constrained examples
- **Expected Impact**: 90%+ format compliance

### Future Work

**6. Multi-Distribution Curriculum Learning**
- Phase 1: Alpaca (learn instruction-following)
- Phase 2: Mixed (Alpaca 60% + Dolly 20% + OpenOrca 20%)
- **Expected Impact**: Robust generalization

**7. Fact-Checking Layer**
- Implement retrieval-augmented generation for knowledge tasks
- Add confidence scoring for factual claims
- **Expected Impact**: Reduce hallucinations by 70%+

---

## Deployment Decision Matrix

| Use Case | Current Model | Recommended Action |
|----------|---------------|-------------------|
| **Alpaca-style chatbot** | Ready | Deploy with length constraints |
| **General assistant** | Limited | Deploy with disclaimers on complex tasks |
| **Technical/factual QA** | Not ready | Block - high hallucination risk |
| **Multi-step reasoning** | Not ready | Block - requires model upgrade |
| **Production API** | Not ready | Block - OOD failure rate too high |

---

## Conclusion

**Success**: Model successfully learned instruction-following for in-distribution tasks (Alpaca +27.5% METEOR, Dolly +26.4% METEOR).

**Critical Failure**: Catastrophic performance on GPT-4 quality prompts (OpenOrca -92.8% BLEU) due to:
1. Insufficient model scale (1.1B parameters)
2. Training distribution mismatch
3. Catastrophic forgetting of reasoning abilities

**Verdict**: **Not production-ready for general deployment**. Acceptable only for narrow Alpaca-style use cases with explicit limitations disclosed to users.

**Next Steps**:
1. Upgrade to 7-8B parameter model (critical)
2. Implement data mixing with OpenOrca (critical)
3. Increase LoRA rank to r=32 (high priority)
4. Add length constraints and structured output training (high priority)