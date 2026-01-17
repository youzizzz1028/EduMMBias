# IAT (Implicit Association Test) Experiment

This experiment is designed to test the performance of AI models in Implicit Association Tests. It evaluates potential biases by combining student attribute images with positive and negative vocabulary.

---

## Experimental Design

### Process Flow

1. **Attribute Pair Composition**: All possible binary combinations are generated for each student attribute (e.g., Gender: Male/Female).
2. **Image Sample Pairs**: For each combination, 3 pairs of image samples are selected. These samples are identical across all other attributes except for the target attribute.
3. **Vocabulary Binding**: Each test binds positive and negative words to specific attribute options.
4. **Bi-directional Testing**: Each attribute pair undergoes "forward" and "reverse" testing (Total of 54 trials per pair: 27 forward combinations + 27 reverse combinations).

### Scale of Experiment

* **Number of Attribute Combinations**: Generated dynamically based on configuration.
* **Trials per Combination**: 3 pairs × 2 directions × 27 word pairs = 162 trials (utilizing all 27 negative words).
* **Total Trials**: Number of Attribute Combinations × 162 trials.

---

## Directory Structure

```text
src/IAT/
├── iat_experiment.py    # Main experiment code
├── test_iat.py          # Testing script
├── analysis.py          # Test result analysis code
└── README.md            # This document

configs/
├── attribute_pairs.yaml # Attribute definitions
└── IAT_attributes.yaml  # Positive/Negative vocabulary

data/
├── generated/
│   ├── profile_images/  # Student images
│   └── profile_metadata.jsonl  # Metadata for image attributes
└── results/
    └── IAT/             # Experimental results

```

---

## Configuration Requirements

### 1. Attribute Configuration (`configs/attribute_pairs.yaml`)

Defines the dimensions of student attributes and their options:

```yaml
gender: ['male', 'female']
race: ['white', 'black', 'asian', 'hispanic', 'American Indian']
socioeconomic_status: ['low', 'middle', 'high']
hobbies: ['sports and travel', 'arts', 'technology', 'reading', 'gaming']
health_status: ['excellent', 'fair', 'poor']

```

### 2. IAT Vocabulary Configuration (`configs/IAT_attributes.yaml`)

Defines the positive and negative words:

```yaml
positive_words:
  - caress
  - freedom
  - health
  # ... total 25 words

negative_words:
  - abuse
  - crash
  - filth
  # ... total 27 words

```

### 3. Image Data

Images should be located in `data/generated/profile_images/`, with corresponding metadata in `data/generated/profile_metadata.jsonl`.

---

## Usage

### 1. Functionality Test

Test the code logic before running the actual experiment:

```bash
python src/IAT/test_iat.py

```

### 2. Run IAT Experiment

```bash
python src/IAT/iat_experiment.py --model <model_name> [--workers <num_workers>]

```

**Parameters:**

* `--model`: Required. The name of the model to be tested (e.g., "gpt-4-vision-preview").
* `--workers`: Optional. Number of concurrent worker threads. Default is 5.

**Example:**

```bash
python src/IAT/iat_experiment.py --model gpt-4-vision-preview --workers 10

```

### 3. Resume from Breakpoint

The experiment supports checkpointing. If interrupted, re-running the same command will resume from where it left off.

---

## Experimental Results

### Result File Format

Results are saved in `data/results/IAT/iat_results_<model_name>.jsonl`. Each line is a JSON object containing:

```json
{
  "target_attr": "gender",
  "option1": "male",
  "option2": "female",
  "positive_word": "love",
  "negative_word": "hate",
  "direction": "forward",
  "image_path": "data/generated/profile_images/...",
  "image_attributes": {...},
  "profile_id": "unique-id",
  "model_name": "gpt-4-vision-preview",
  "model_response": "Full model response text",
  "choice": "A",
  "reason": "Reason for choice",
  "confidence": 85,
  "success": true
}

```

---

## Experimental Logic

### 1. Attribute Pair Generation

For each attribute dimension, all possible binary combinations are generated:

* **Gender**: Male/Female
* **Race**: White/Black, White/Asian, Black/Hispanic, etc.
* **Socioeconomic Status**: Low/Middle, Low/High, Middle/High.
* **Hobbies & Health**: All pair-wise combinations between defined options.

### 2. Image Matching

Finds 3 pairs of image samples for each attribute pair. These pairs must be identical in all attributes except for the target dimension to ensure controlled variables.

### 3. Test Generation

Each attribute pair generates 54 trials:

* **Forward**: `positive_word + option1` vs `negative_word + option2` (27 trials).
* **Reverse**: `positive_word + option2` vs `negative_word + option1` (27 trials).
* Ensures every `negative_word` is used exactly once per direction.

### 4. Model Invocation

Calls the model using multi-threading with built-in retry mechanisms and error handling.

---

## Troubleshooting

* **Image file not found**: Check the `data/generated/profile_images/` directory.
* **API call failure**: Verify your API key, quotas, and network connection.
* **Out of memory**: Reduce the number of `--workers`.
* **Corrupted results**: Delete the corrupted `.jsonl` file and restart to trigger the resume logic.