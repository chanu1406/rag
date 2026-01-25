# Local Brain - Environment Setup Script
# This script sets up a Python 3.11 virtual environment

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "Local Brain - Environment Setup" -ForegroundColor Cyan
Write-Host "======================================`n" -ForegroundColor Cyan

# Check for Python 3.11 or 3.12
Write-Host "Checking for Python 3.11/3.12..." -ForegroundColor Yellow

$pythonVersions = py -0 2>&1 | Out-String
$hasPython311 = $pythonVersions -match "3\.11"
$hasPython312 = $pythonVersions -match "3\.12"

if (-not $hasPython311 -and -not $hasPython312) {
    Write-Host "`n[ERROR] Python 3.11 or 3.12 not found!" -ForegroundColor Red
    Write-Host "`nPython 3.13 is too new for the AI/ML ecosystem." -ForegroundColor Yellow
    Write-Host "Please install Python 3.11:" -ForegroundColor Yellow
    Write-Host "  1. Visit: https://www.python.org/downloads/release/python-3119/" -ForegroundColor Cyan
    Write-Host "  2. Download 'Windows installer (64-bit)'" -ForegroundColor Cyan
    Write-Host "  3. Run installer and check 'Add Python to PATH'" -ForegroundColor Cyan
    Write-Host "  4. Re-run this script`n" -ForegroundColor Cyan
    exit 1
}

# Determine which version to use
$pythonVersion = if ($hasPython311) { "3.11" } else { "3.12" }
Write-Host "[OK] Found Python $pythonVersion`n" -ForegroundColor Green

# Create virtual environment
$venvPath = ".venv"
if (Test-Path $venvPath) {
    Write-Host "Virtual environment already exists at $venvPath" -ForegroundColor Yellow
    $response = Read-Host "Delete and recreate? (y/N)"
    if ($response -eq "y" -or $response -eq "Y") {
        Remove-Item -Recurse -Force $venvPath
        Write-Host "Deleted existing venv`n" -ForegroundColor Yellow
    } else {
        Write-Host "Using existing venv`n" -ForegroundColor Green
        & "$venvPath\Scripts\Activate.ps1"
        exit 0
    }
}

Write-Host "Creating virtual environment with Python $pythonVersion..." -ForegroundColor Yellow
py -$pythonVersion -m venv $venvPath

if (-not $?) {
    Write-Host "[ERROR] Failed to create virtual environment" -ForegroundColor Red
    exit 1
}

Write-Host "[OK] Virtual environment created`n" -ForegroundColor Green

# Activate venv
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& "$venvPath\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "`nUpgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install PyTorch with CUDA
Write-Host "`nInstalling PyTorch with CUDA 12.1..." -ForegroundColor Yellow
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install requirements
Write-Host "`nInstalling project dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

Write-Host "`n======================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "======================================`n" -ForegroundColor Cyan

Write-Host "Virtual environment activated. To activate it manually in the future:" -ForegroundColor Yellow
Write-Host "  .venv\Scripts\Activate.ps1`n" -ForegroundColor Cyan

Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Install Ollama: https://ollama.com/download" -ForegroundColor Cyan
Write-Host "  2. Pull model: ollama pull llama3" -ForegroundColor Cyan
Write-Host "  3. Test ingestion: python main.py ingest ./data/documents" -ForegroundColor Cyan
Write-Host "  4. Start chat: python main.py chat`n" -ForegroundColor Cyan
