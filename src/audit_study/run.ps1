$models = @(
    # "gpt-4o"，
    # "gpt-5.2",
    "gemini-3-flash-preview-thinking",
    "grok-4-fast-reasoning"
)

foreach ($model in $models) {
    Write-Host "Test Model: $model" -ForegroundColor Cyan
    python src/audit_study/vlm_test.py --model $model
}
Write-Host "All tasks completed" -ForegroundColor Green