param(
    [string]$BindHost = "0.0.0.0",
    [int]$Port = 8000,
    [ValidateSet("glb", "obj")]
    [string]$OutputFormat = "glb",
    [string]$ImagesDir,
    [string]$ApiKey,
    [int]$MaxImages = 24,
    [int]$HealthTimeoutSeconds = 60,
    [int]$HealthPollMs = 500,
    [switch]$SkipValidation,
    [switch]$KeepBackendRunning
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Stage {
    param([string]$Message)
    Write-Host ("[runbook] " + $Message)
}

function Get-DotEnvValue {
    param(
        [string]$DotEnvPath,
        [string]$Name
    )

    if (-not (Test-Path -LiteralPath $DotEnvPath)) {
        return $null
    }

    $pattern = "^\s*$([regex]::Escape($Name))\s*=\s*(.*)\s*$"
    foreach ($line in Get-Content -LiteralPath $DotEnvPath) {
        if ($line -match "^\s*#") {
            continue
        }
        if ($line -match $pattern) {
            $value = $Matches[1].Trim()
            if (
                ($value.StartsWith('"') -and $value.EndsWith('"')) -or
                ($value.StartsWith("'") -and $value.EndsWith("'"))
            ) {
                if ($value.Length -ge 2) {
                    $value = $value.Substring(1, $value.Length - 2)
                }
            }
            return $value
        }
    }

    return $null
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "No existe $pythonExe. Crea/activa el virtualenv antes de ejecutar este script."
}

Set-Location -LiteralPath $repoRoot

$baseUrl = "http://$BindHost`:$Port"
$stdoutLog = Join-Path $repoRoot "uvicorn_auto_stdout.log"
$stderrLog = Join-Path $repoRoot "uvicorn_auto_stderr.log"
$dotEnvPath = Join-Path $repoRoot ".env"

$effectiveApiKey = $ApiKey
if ([string]::IsNullOrWhiteSpace($effectiveApiKey) -and -not [string]::IsNullOrWhiteSpace($env:LOCAL3D_API_KEY)) {
    $effectiveApiKey = $env:LOCAL3D_API_KEY
}
if ([string]::IsNullOrWhiteSpace($effectiveApiKey)) {
    $effectiveApiKey = Get-DotEnvValue -DotEnvPath $dotEnvPath -Name "LOCAL3D_API_KEY"
}

$requestHeaders = @{}
if (-not [string]::IsNullOrWhiteSpace($effectiveApiKey)) {
    $requestHeaders["X-API-Key"] = $effectiveApiKey
    Write-Stage "API key detectada: se enviara header X-API-Key en validaciones."
}

$uvicornArgs = @(
    "-m", "uvicorn",
    "main:app",
    "--host", $BindHost,
    "--port", $Port.ToString()
)

Write-Stage "Iniciando backend en $baseUrl"
$backendProcess = Start-Process `
    -FilePath $pythonExe `
    -ArgumentList $uvicornArgs `
    -PassThru `
    -WorkingDirectory $repoRoot `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog

$healthOk = $false
$healthPayload = $null
$attempts = [Math]::Max(1, [int]([Math]::Ceiling(($HealthTimeoutSeconds * 1000.0) / $HealthPollMs)))

try {
    for ($i = 0; $i -lt $attempts; $i++) {
        if ($backendProcess.HasExited) {
            throw "El backend termino antes de responder /health. Revisa $stderrLog"
        }

        try {
            $healthPayload = Invoke-RestMethod -Uri "$baseUrl/health" -Method GET -Headers $requestHeaders -TimeoutSec 2
            if ($healthPayload.status -eq "ok") {
                $healthOk = $true
                break
            }
        }
        catch {
            # Continue polling until timeout.
        }

        Start-Sleep -Milliseconds $HealthPollMs
    }

    if (-not $healthOk) {
        throw "Timeout esperando /health en $baseUrl. Revisa $stderrLog"
    }

    Write-Stage "Health OK. engine=$($healthPayload.engine)"
    if ($healthPayload.colmap) {
        Write-Stage ("COLMAP config: use_gpu={0} enable_dense_stages={1} require_dense_reconstruction={2}" -f `
            $healthPayload.colmap.use_gpu, `
            $healthPayload.colmap.enable_dense_stages, `
            $healthPayload.colmap.require_dense_reconstruction)
    }

    if ($SkipValidation) {
        Write-Stage "Validacion omitida por parametro -SkipValidation."
        exit 0
    }

    $e2eArgs = @(
        "tests\run_real_colmap_e2e.py",
        "--base-url", $baseUrl,
        "--output-format", $OutputFormat
    )

    if ($ImagesDir) {
        $resolvedImagesDir = (Resolve-Path -LiteralPath $ImagesDir).Path
        $e2eArgs += @("--images-dir", $resolvedImagesDir)
    }

    if ($MaxImages -gt 0) {
        $e2eArgs += @("--max-images", $MaxImages.ToString())
    }
    if (-not [string]::IsNullOrWhiteSpace($effectiveApiKey)) {
        $e2eArgs += @("--api-key", $effectiveApiKey)
    }

    Write-Stage "Ejecutando validacion E2E real..."
    & $pythonExe @e2eArgs
    $e2eCode = $LASTEXITCODE
    if ($e2eCode -ne 0) {
        throw "La validacion E2E termino con codigo $e2eCode."
    }

    Write-Stage "Validacion E2E exitosa."
}
finally {
    if ($backendProcess -and -not $backendProcess.HasExited -and -not $KeepBackendRunning) {
        Write-Stage "Deteniendo backend (sin -KeepBackendRunning)."
        Stop-Process -Id $backendProcess.Id -Force
    }
    elseif ($backendProcess -and -not $backendProcess.HasExited -and $KeepBackendRunning) {
        Write-Stage "Backend sigue activo con PID=$($backendProcess.Id)."
    }
}
