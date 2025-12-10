# Named Entity Recognition (NER) Scenario

## Overview

Named Entity Recognition (NER) is a fundamental Natural Language Processing (NLP) task that involves identifying and classifying named entities present in unstructured text into predefined categories. These categories typically include:

-   **Person (PER)**: Names of people.
-   **Organization (ORG)**: Companies, institutions, government bodies.
-   **Location (LOC)**: Cities, countries, rivers, etc.
-   **Miscellaneous (MISC)**: Other entities like events, works of art, etc.

### Significance in LLMs

NER is a critical capability for Large Language Models (LLMs) for several reasons:

1.  **Information Extraction**: It allows LLMs to structure unstructured data, turning raw text into actionable insights.
2.  **Knowledge Graph Construction**: Identifying entities is the first step in building relationships between them.
3.  **Search and Indexing**: Improves search relevance by understanding the specific entities users are looking for.
4.  **Context Understanding**: Helps the model understand the "who", "what", and "where" of a text passage.

## Implementation in FMBench

In FMBench, the NER scenario is implemented to evaluate an LLM's ability to accurately extract specific entities from a given text.

### Implementation Details

-   **Class**: The core logic is implemented in the `NERScenario` class within `software/components/scenarios/common_nlp_scenarios.py`.
-   **Dataset**: By default, it uses the **Wikiann** dataset (multilingual named entity recognition), but this is configurable.
-   **Prompt Engineering**: The scenario uses a prompt template to instruct the model.
    -   *Default Template*: `"Extract named entities from the following text: {input}"`
    -   *Input*: A sentence from the dataset.
    -   *Target Output*: A formatted list of entities, e.g., `"Barack Obama (PER), USA (LOC)"`.

### How it Works

1.  **Data Processing**:
    -   The system reads tokenized data (list of words) and their corresponding NER tags (e.g., `B-PER`, `I-ORG`) from the dataset.
    -   It reconstructs the sentence from tokens.
    -   It processes the correct tags to form a "ground truth" string in the format: `"Entity Name (Type)"`.

2.  **Model Interaction**:
    -   The reconstructed sentence is formatted with the prompt template.
    -   The LLM generates a response listing the entities it found.

3.  **Metric Computation**:
    -   **Accuracy (Recall-based)**: The system parses the model's output to extract entity names.
    -   It compares these found entities against the expected entities (ground truth).
    -   The score is calculated as: `(Number of Correctly Retrieved Entities) / (Total Expected Entities)`.

## Running the Scenario

To run the NER scenario, you use the `ner` configuration.

### Configuration

The configuration is defined in `software/conf/scenario/ner.yaml`:

```yaml
name: ner
_target_: components.scenarios.common_nlp_scenarios.NERScenario
dataset_name: wikiann
dataset_config: en
split: test
input_key: tokens
target_key: ner_tags
tag_map:
  0: "O"
  1: "B-PER"
  2: "I-PER"
  3: "B-ORG"
  4: "I-ORG"
  5: "B-LOC"
  6: "I-LOC"
num_samples: 20
prompt_template: "Perform Named Entity Recognition on the following text. List entities as 'Entity (Type)'.\n\nText: {input}\n\nEntities:"
```

### Execution

This scenario is typically executed as part of a benchmarking run. You can include it in your main configuration file by referencing `scenario: ner`.
