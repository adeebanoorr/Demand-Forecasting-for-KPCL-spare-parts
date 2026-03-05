# KPCL Forecasting App - Run Full ML Pipeline
# Run from project root: .\run_pipeline.ps1

$PROJECT_ROOT = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $PROJECT_ROOT

function Invoke-Step {
    param([string]$Label, [string]$Script)
    Write-Host ""
    Write-Host "--------------------------------------------" -ForegroundColor Cyan
    Write-Host "  $Label" -ForegroundColor Cyan
    Write-Host "--------------------------------------------" -ForegroundColor Cyan
    python $Script
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: $Label failed (exit code $LASTEXITCODE)" -ForegroundColor Red
        Write-Host "Pipeline stopped. Fix the error and re-run." -ForegroundColor Red
        exit $LASTEXITCODE
    }
    Write-Host "  DONE: $Label" -ForegroundColor Green
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  KPCL Forecasting - Full ML Pipeline" -ForegroundColor Cyan
Write-Host "  Started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
Write-Host "========================================" -ForegroundColor Cyan

Invoke-Step "Step 1/6: Data Preparation"                  "src/data/data_preparation.py"
Invoke-Step "Step 2/6: Classic ML Model Comparison"       "src/modeling/compare_classic_ml_rmse.py"
Invoke-Step "Step 3/6: Time Series Model Comparison"      "src/modeling/compare_models_rmse.py"
Invoke-Step "Step 4/6: Auto-SARIMA Training"              "src/modeling/train_forecast_autosarima.py"
Invoke-Step "Step 5/6: Final Forecast - All Models"       "src/modeling/train_forecast_all_models.py"
Invoke-Step "Step 6/6: Validation - All Models"           "src/forecast_validation/validate_all_models.py"

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  PIPELINE COMPLETE!" -ForegroundColor Green
Write-Host "  Finished: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
Write-Host "  Now run .\start.ps1 to launch the app." -ForegroundColor White
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
