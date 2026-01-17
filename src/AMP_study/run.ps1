$models = @(
    "grok-4-fast-reasoning"
    # "gemini-3-flash-preview",
    # "gpt-5.2",
    # "claude-sonnet-4-5-20250929"
)

foreach ($model in $models) {
    Write-Host "Test Model: $model" -ForegroundColor Cyan
    python src/AMP_study/AMP_experiment.py --model $model
}
Write-Host "All tasks completed" -ForegroundColor Green