# Edu-MMBias
## Framework

![Framework Overview](figure/framework.png)

## Project Overview

This project detects potential biases in large language models (LLMs) in educational applications based on the three-component model of attitudes (cognitive, affective, behavioral):

1. **Cognitive Component** - Measured via Implicit Association Test (IAT) to assess automatic concept associations
2. **Affective Component** - Measured via Affect Misattribution Procedure (AMP) to assess emotional responses
3. **Behavioral Component** - Measured via Audit Study to assess bias in decision-making tasks

## Theoretical Basis

### Three-Component Model of Attitudes
1. **Cognitive Component**: Reflects beliefs and knowledge about the attitude object, measured by IAT
2. **Affective Component**: Reflects emotional reactions to the attitude object, measured by AMP  
3. **Behavioral Component**: Reflects behavioral tendencies toward the attitude object, measured by Audit Study
## Quick Start

### Installation
```bash
pip install openai pyyaml
```

### Configuration
Edit `configs/model_config.yaml`:
```yaml
models:
  gpt-5:
    reason: "mix"
    provider: "OpenAI"
    model_name: "gpt-5-2025-08-07"

default:
  base_url: "https://your-api-endpoint/v1"
  api_key: "your-api-key-here"
  log: "llm_call.log"
  max_retries: 3
  retry_delay: 1
  timeout: 60
```
## Experiment Instructions

### 1. Implicit Association Test (IAT)
```bash
cd src/IAT
./run.ps1
```
- Configuration: `configs/IAT_attributes.yaml`
- Measures automatic concept associations (cognitive component)
- Results saved: `data/results/iat_results.jsonl`

### 2. Affect Misattribution Procedure (AMP)
```bash
./src/AMP_study/run.ps1
```
- Requires neutral image pool: `data/raw/neutral_images/`
- Measures emotional responses to priming stimuli (affective component)
- Results saved: `data/results/amp_results.jsonl`

### 3. Audit Study
```bash
./src/audit_study/run.ps1
```
- Configuration: `configs/attribute_pairs.yaml`
- Measures bias in decision-making tasks (behavioral component)
- Results saved: `data/results/audit_results.jsonl`
## Project Structure

```text
.
├── configs/                        # Configuration files
│   ├── model_config.yaml           # Model configurations
│   ├── attribute_pairs.yaml        # Student attribute pairs
│   └── IAT_attributes.yaml         # IAT test attributes
├── src/                            # Source code
│   ├── AMP_study/                  # AMP study implementation
│   │   ├── AMP_experiemnt.py       # AMP experiment
│   │   └── AMP_neutral_track.py    # Neutral image processing
│   ├── audit_study/                # Audit study implementation
│   │   ├── analysis.py             # Analysis tools
│   │   ├── audit_experiment.py     # Audit experiment
│   │   ├── README.md               # Documentation
│   │   ├── run.ps1                 # Run script
│   │   ├── test_simple.py          # Simple tests
│   │   └── text_only.py            # Text-only experiments
│   ├── data_generation/            # Data generation
│   │   ├── AMP_neutral_pic.py      # AMP neutral images
│   │   ├── audit_profiles.py       # Audit study profiles
│   │   ├── confirm_and_replace_images.py # Image confirmation
│   │   ├── generate_iat_questions.py # IAT questions
│   │   ├── profile.py              # Student profiles
│   │   ├── regenerate_missing_original_images.py # Regenerate missing images
│   │   └── regenerate_problematic_images.py # Regenerate problematic images
│   ├── IAT/                        # IAT implementation
│   │   ├── analysis.py             # Analysis tools
│   │   ├── iat_experiment.py       # IAT experiment
│   │   ├── README.md               # Documentation
│   │   ├── run.ps1                 # Run script
│   │   └── test_iat.py             # IAT tests
│   └── model_api/                  # Model API clients
│       ├── api_utils.py            # API utilities
│       ├── client_openai.py        # OpenAI-compatible client
│       ├── image_generation.py     # Image generation API
│       └── multimodal_openai.py    # Multimodal API
├── tests/                          # Tests
│   └── test_model_config_models.py # Model config tests
├── data/                           # Data
│   ├── raw/                        # Raw data
│   ├── generated/                  # Generated data
│   └── results/                    # Analysis results
├── figure/                         # Figures
│   ├── amp_forest_plot.png         # AMP results
│   ├── audit_results.png           # Audit study results
│   ├── framework.png               # Project framework
│   ├── IAT_results.png             # IAT results
│   ├── pipeline.png                # Research pipeline
│   └── sunburst.png                # Data distribution
└── README.md                       # Project documentation
```





## Contribution
1. Fork project
2. Create feature branch
3. Submit Pull Request

## License
MIT License