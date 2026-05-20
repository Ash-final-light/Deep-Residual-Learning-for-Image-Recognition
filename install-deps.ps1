param(
    [string]$Python = "E:\PythonAlgorithm2\.venv\Scripts\python.exe",
    [switch]$Offline
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Wheelhouse = Join-Path $ProjectRoot "wheelhouse"
$Requirements = Join-Path $ProjectRoot "requirements-cpu.txt"

if (-not (Test-Path $Python)) {
    throw "Python interpreter not found: $Python"
}

if (-not (Test-Path $Requirements)) {
    throw "Requirements file not found: $Requirements"
}

& $Python --version
& $Python -m pip --version

if ($Offline) {
    if (-not (Test-Path $Wheelhouse)) {
        throw "Offline wheelhouse not found: $Wheelhouse"
    }
    & $Python -m pip install --no-index --find-links $Wheelhouse -r $Requirements
} else {
    & $Python -m pip install `
        -r $Requirements `
        --extra-index-url https://download.pytorch.org/whl/cpu `
        -i https://pypi.tuna.tsinghua.edu.cn/simple `
        --trusted-host pypi.tuna.tsinghua.edu.cn `
        --trusted-host download.pytorch.org
}

& $Python -c "import torch, numpy, sklearn, pandas, tqdm; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('numpy', numpy.__version__); print('sklearn', sklearn.__version__); print('pandas', pandas.__version__); print('tqdm', tqdm.__version__)"