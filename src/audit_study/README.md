# VLM Audit Study Testing Module

## Overview

This module is designed to execute VLM (Vision-Language Model) testing within educational bias audit research. It records the selection results of VLM models based on two provided student images and a specific question.

## Key Features

* **Multimodal Support**: Supports VLM models with image input capabilities (GPT-4o, Claude Sonnet, Gemini Flash, etc.).
* **Automated Sample Generation**: Automatically generates test sample pairs based on attribute combinations.
* **Batch Testing**: Supports large-scale batch testing (up to 4,050 test cases).
* **Result Logging**: Automatically records test results into JSONL files.
* **Error Handling**: Comprehensive error handling and retry mechanisms.
* **Progress Display**: Real-time progress tracking using `tqdm`.
* **Multithreaded Acceleration**: Supports concurrent execution to significantly boost testing speed.
* **Structured Output**: Requires VLMs to respond in a specific format to facilitate subsequent statistical analysis.
* **Statistical Analysis**: Provides statistical methods including mean, variance, T-tests, and forest plots.

## File Structure

```
src/audit_study/
├── __init__.py          # Module initialization
├── audit_experiment.py  # Main testing logic
├── test_simple.py       # Simple testing script
├── analysis.py          # Statistical analysis module
├── text_only.py         # Text-only modality testing module
└── README.md            # This documentation

```

## Dependencies

Ensure the following Python packages are installed:

```bash
pip install openai pyyaml tqdm

```

## Usage

### 1. Run the Main Module Directly

```bash
python src/audit_study/audit_experiment.py

```

### 2. Usage in Code

```python
from src.audit_study import AuditStudy

# Initialize the study
study = AuditStudy()
study.load_configs()

# Run tests
results = study.run_test(model_name="gpt-4o", output_file="results.jsonl")

```

## Configuration

### Supported VLM Models

* `gpt-4o`: OpenAI GPT-4o (Default)
* `gpt-4.1`: OpenAI GPT-4.1
* `claude-sonnet`: Anthropic Claude Sonnet
* `gemini-flash`: Google Gemini Flash

### Testing Parameters

* `samples_per_pair`: Number of sample pairs per attribute combination (Default: 3).
* `output_file`: Path for the output file (Default: `vlm_test_results.jsonl`).
* `max_retries`: Maximum number of API retries (Default: 3).
* `timeout`: API call timeout in seconds (Default: 60s).

## Data Workflow

1. **Load Configuration**: Read attribute combination settings (`configs/attribute_pairs.yaml`).
2. **Load Data**: Read student image metadata (`data/generated/profile_metadata.jsonl`) and task scenarios (`data/raw/scenario_en_50_final.json`).
3. **Generate Sample Pairs**: Create sample pairs for each attribute combination (27 combinations × 3 samples = 81 pairs).
4. **Combine Tests**: Combine each sample pair with 50 scenarios (81 × 50 = 4,050 test cases).
5. **Call VLM**: Provide two student images and a question to the VLM model.
6. **Record Results**: Save VLM selection results to a JSONL file.

## Output Format

Test results are saved in JSONL format. Each record contains full test information and parsed structured data:

### Full Record Format

```json
{
  "test_id": "gender_male_female_0_scenario_1",
  "sample_pair": {
    "pair_id": "gender_male_female_0",
    "attribute": "gender",
    "value1": "male",
    "value2": "female",
    "base_attributes": {...},
    "profile1": {...},
    "profile2": {...}
  },
  "scenario": {
    "id": 1,
    "question": "...",
    "concepts": [...]
  },
  "vlm_response": "Task A: Student A\nTask B: Student B\nReason: Student A seems better suited for positive tasks...",
  "parsed_response": {
    "task_a_assignment": "Student A",
    "task_b_assignment": "Student B",
    "reason": "Student A seems better suited for positive tasks...",
    "parsing_successful": true
  },
  "model_name": "gpt-4o"
}

```

### Structured Output Requirements

VLMs are instructed to respond strictly in the following format:

```
Task A: [Student A/Student B]
Task B: [Student A/Student B]
Reason: [Brief explanation of assignment logic]

```

### Parsed Data Structure

* `task_a_assignment`: Which student was assigned Task A (Student A/Student B).
* `task_b_assignment`: Which student was assigned Task B (Student A/Student B).
* `reason`: The text of the assignment reasoning.
* `parsing_successful`: Boolean indicating if parsing was successful.

### Statistical Analysis Benefits

Structured output simplifies subsequent statistical analysis:

* Easily count assignment preferences for each attribute combination.
* Analyze VLM biases toward different student attributes.
* Facilitate quantitative fairness evaluations.
* Support automated data analysis and visualization.

## Verification

Run the simple test script to verify module functionality:

```bash
python src/audit_study/test_simple.py

```

## Important Notes

1. **API Keys**: Ensure the correct API keys are configured in `configs/model_config.yaml`.
2. **Image Paths**: Image file paths in `profile_metadata.jsonl` are relative; the code will automatically convert them to absolute paths.
3. **Network Connection**: Testing requires a stable internet connection to call the VLM APIs.
4. **Cost Consideration**: Large-scale testing may incur significant API fees; it is recommended to conduct small-scale tests first.

## Troubleshooting

### Common Issues

1. **Image File Missing**: Check if the `data/generated/profile_images/` directory contains the necessary images.
2. **API Call Failed**: Verify your API key and network connectivity.
3. **Out of Memory**: Large-scale tests may require substantial memory; consider processing in batches.

### Log Files

Logs during the testing process are saved in the `vlm_test.log` file for debugging purposes.

---