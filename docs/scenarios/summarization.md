# Summarization Scenario

## Overview

Text Summarization is an NLP task that involves creating a concise and fluent summary of a longer text document while retaining its most important information. There are two main types:
-   **Extractive Summarization**: Selects key sentences from the original text.
-   **Abstractive Summarization**: Generates new sentences that capture the core meaning, often paraphrasing the original content.

### Significance in LLMs

Summarization is a flagship capability for Large Language Models (LLMs) because:
1.  **Efficiency**: It helps users quickly digest large volumes of information (news, reports, papers).
2.  **Content Generation**: It demonstrates the model's deep understanding of context, nuance, and hierarchy of importance within a text.
3.  **Cross-domain Utility**: It is applicable in legal, medical, technical, and general domains.

## Implementation in FMBench

In FMBench, the Summarization scenario measures an LLM's ability to generate meaningful abstracts for news articles or other long-form text.

### Implementation Details

-   **Class**: The logic is encapsulated in the `SummarizationScenario` class within `software/components/scenarios/common_nlp_scenarios.py`.
-   **Dataset**: The default configuration uses the **CNN/DailyMail** dataset (version 3.0.0).
-   **Prompt Engineering**:
    -   *Default Template*: `"Summarize the following article.\n\nArticle: {input}\n\nSummary:"`
    -   *Input*: The full text of an article.
    -   *Target Output*: The "highlights" or human-written summary associated with the article.

### How it Works

1.  **Data Processing**:
    -   The system iterates through the dataset.
    -   It extracts the input text (e.g., from the `article` key) and the ground truth summary (e.g., from the `highlights` key).
    -   It formats the input using the prompt template.

2.  **Model Interaction**:
    -   The LLM generates a summary based on the provided article.

3.  **Metric Computation**:
    -   The generated summary is compared against the ground truth using standard NLP metrics:
        -   **BLEU**: Measures n-gram overlap.
        -   **METEOR**: Aligns words using exact, stem, synonym, and paraphrase matches (requires NLTK).
        -   **BERTScore**: Computes similarity using contextual embeddings. *Note: This is resource-intensive and can be disabled via `use_expensive_metrics: false` configuration.*

## Running the Scenario

To run the Summarization scenario, you use the `summarization` configuration.

### Configuration

The configuration is defined in `software/conf/scenario/summarization.yaml`:

```yaml
name: summarization
_target_: components.scenarios.common_nlp_scenarios.SummarizationScenario
dataset_name: cnn_dailymail
dataset_config: "3.0.0"
split: test
input_key: article
target_key: highlights
num_samples: 20
prompt_template: "Summarize the following article.\n\nArticle: {input}\n\nSummary:"
use_expensive_metrics: false
```

### Execution

Include `scenario: summarization` in your main benchmark configuration file to execute this test.
