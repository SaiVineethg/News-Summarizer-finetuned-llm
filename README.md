# News Summarizer Multi Language — Llama 3.2 Fine-Tuning   

A fine-tuned **Llama 3.2 1B Instruct** model for converting full news articles into concise **40–60 word news summaries**, similar to short-news applications.

The project also experiments with **out-of-scope request handling**, where non-news inputs such as mathematics, coding, recipes, travel planning, and general questions should be rejected.

## Project Goal

```text
Full News Article
       ↓
Fine-tuned Llama 3.2 1B
       ↓
40–50 Word Summary
```

For non-news requests:

```text
Non-News Request
       ↓
"I can only summarize news articles."
```

The main objectives are:

- Main event extraction
- Important facts, names, dates, numbers, and outcomes
- Concise news writing
- 40–60 word output control
- Factual summarization
- Out-of-scope request handling

## Model

**Base model:** Llama 3.2 1B Instruct

**Fine-tuning stack:**

- Unsloth
- LoRA / QLoRA
- 4-bit quantized base model
- Llama 3 chat template


## Fine-Tuning Configuration

Initial LoRA configuration:

```python
r = 32

target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

lora_alpha = 64
lora_dropout = 0
bias = "none"
use_gradient_checkpointing = "unsloth"
random_state = 3407
use_rslora = False
```

Model loading:

```python
max_seq_length = 1024*10
dtype = None
load_in_4bit = True
```

## Dataset

The dataset uses JSONL format.

### In-scope news categories

The model is trained to summarize:

- Politics
- Business
- Technology
- Sports
- Entertainment
- Science
- World
- Health

Example:

```json
{
  "article": "The central bank announced...",
  "reference_summary": "The central bank kept its benchmark interest rate unchanged...",
  "category": "business",
  "difficulty": "medium"
}
```

### Out-of-scope categories

The second training iteration adds:

- Mathematics
- Coding
- General knowledge
- Translation
- Creative writing
- Recipes
- Personal advice
- Travel planning
- Definitions
- Trivia
- Weather
- Programming
- Grammar

Example:

```json
{
  "article": "What is 125 multiplied by 24?",
  "reference_summary": "I can only summarize news articles.",
  "category": "out_of_scope",
  "difficulty": "easy",
  "topic": "Mathematics"
}
```

## Expected Behavior

### News input

```text
The central bank announced that it would keep its benchmark interest rate unchanged amid persistent inflation concerns...
```

Expected behavior: a factual summary containing the most important information and approximately **40–50 words**.

### Non-news input

```text
What is 125 × 24?
```

Expected:

```text
I can only summarize news articles.
```

## Training Pipeline

```text
JSONL Dataset
     ↓
Train / Validation / Test Split
     ↓
Llama 3 Chat Formatting
     ↓
4-bit Base Model
     ↓
LoRA Adapters
     ↓
SFTTrainer
     ↓
Response-Only Loss
     ↓
Fine-Tuned Model
     ↓
Sanity Testing
     ↓
GGUF Export
     ↓
Ollama
```

## Chat Formatting

The JSONL records are converted to the Llama 3 chat format:

```text
<|begin_of_text|>
<|start_header_id|>user<|end_header_id|>

[Instruction + Article]
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

[Reference Summary]
<|eot_id|>
```

`reference_summary` is the supervised target.

## Core Prompt

```text
You are a news summarization assistant. Your only task is to summarize news articles in 40-50 words. Keep summaries factual, concise, and focused on the most important information. Do not add information that is not present in the article. If the input is not a news article, respond exactly with "I can only summarize news articles."
```

## Evaluation

### News summarization

Evaluate:

- 40–60 word compliance
- Relevance
- Factuality
- Conciseness
- Readability
- Main-event coverage


Also consider semantic similarity and LLM/human factuality evaluation.

### Out-of-scope handling

Measure the percentage of non-news requests for which the model returns the expected refusal.

Examples:

```text
1 + 1
Write Python code to reverse a string.
Translate "Hello" into Telugu.
What is the capital of Australia?
Give me a recipe for fried rice.
Plan a three-day trip to Goa.
What does inflation mean?
What will the weather be tomorrow?
Write a poem about rain.
```

Expected response:

```text
I can only summarize news articles.
```

## Training and Deployment Environment

The model fine-tuning was performed on **Google Colab** rather than on the local machine.

Training workflow:

```text
Google Colab
    ↓
Llama 3.2 1B Instruct
    ↓
QLoRA / Unsloth fine-tuning
    ↓
Fine-tuned model
    ↓
GGUF export
```

After fine-tuning, the exported **GGUF model was downloaded to the local machine** and loaded into the **Ollama Desktop application** for local inference and testing.

Deployment workflow:

```text
Google Colab
    ↓
Fine-tuning
    ↓
GGUF export
    ↓
Local Machine
    ↓
Ollama Desktop
    ↓
Local inference
```

The local machine is therefore used for **model serving and testing**, while the computationally intensive fine-tuning process is performed in Google Colab.

## Ollama Deployment

The fine-tuned model can be exported to GGUF and loaded into Ollama.

Example:

```text
FROM short_news_v2_llama-3.2-1b-instruct.Q4_K_M.gguf
```

A system prompt can reinforce the intended behavior:

```text
SYSTEM """
You are ONLY a news summarization assistant.

Your only allowed task is:
NEWS ARTICLE -> 40-50 WORD NEWS SUMMARY

For any input that is not a news article, respond exactly:
I can only summarize news articles.

Do not answer mathematics, coding, programming, translation,
general knowledge, recipes, travel planning, trivia, grammar,
weather, creative writing, or personal advice requests.
"""
```

Recommended generation settings for factual summarization:

```text
temperature 0.0 - 0.3
```

Example:

```bash
ollama create News-summary -f Modelfile
ollama run News-summary
```

## Lessons Learned

### Chat templates matter

Training and inference should use the same chat template. Llama 3 uses markers such as:

```text
<|start_header_id|>user<|end_header_id|>
<|start_header_id|>assistant<|end_header_id|>
<|eot_id|>
```

Using incompatible markers such as Qwen-style `<|im_start|>` can cause response-masking failures.

### Response-only training

`train_on_responses_only()` is intended to calculate loss mainly on the assistant response rather than the user/article portion. A mismatch between the configured response marker and the tokenizer template can result in all labels being `-100`, meaning there is no usable training signal.

### Fine-tuning does not erase base-model capabilities

Fine-tuning can teach the model new task-specific behavior, but the base model may still retain capabilities such as arithmetic, coding, and general question answering. Out-of-scope examples and system instructions can improve specialization, but they are not a hard capability firewall.

## Recommended Production Architecture

For robust production behavior, separate routing from summarization:

```text
User Input
    ↓
News / Non-News Check
    ↓
 ┌───────────────┐
 │               │
News          Non-News
 │               │
 ↓               ↓
Fine-tuned     Fixed response
Llama
 │
 ↓
40–60 word summary
```

A small classifier or application-level gate can prevent unrelated requests from reaching the summarization model.

## Suggested Repository Structure

```text
News_summary/
│
├── data/
│   ├── news_short_summary.jsonl
│   └── out_of_scope.jsonl
│
├── notebooks/
│   └── llama3_news_finetuning.ipynb
│
├── ollama/
│   └── Modelfile
│
├── models/
│   └── *.gguf
│
├── outputs/
│
├── requirements.txt
│
└── README.md
```

## Future Improvements

- Increase the amount and quality of news training data
- Improve reference-summary quality with human/editorial review
- Add longer and more difficult news articles
- Expand out-of-scope examples
- Add a dedicated news/non-news classifier
- Add automatic 40–50 word validation

## Disclaimer

This is an experimental fine-tuning project for learning and evaluation. Generated summaries should be checked for factual accuracy before being used in a production news application.

## License

Add the license appropriate for this repository and verify the licenses of all datasets, pretrained models, and third-party assets before redistribution.


## Author

**Sai Vineeth G**

📧 Email: saivineethgaddam@gmail.com
