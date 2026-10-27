#Requires -Version 5.1
<#
.SYNOPSIS
    Publishes the local WebAssembly Release build of FuchsUndHenne directly to the
    'gh-pages' branch on GitHub Pages without pushing training checkpoints or build caches.

.PARAMETER RemoteUrl
    Git remote URL for pushing the gh-pages branch. Defaults to origin remote if configured,
    or https://github.com/mesw/FuchsUndHenne.git.

.PARAMETER Branch
    Target deployment branch name (default: gh-pages).
#>

[CmdletBinding()]
param(
    [string]$RemoteUrl = "",
    [string]$Branch = "gh-pages"
)

$ErrorActionPreference = "Stop"

$repoRoot = $PSScriptRoot

# 0. Locate WebAssembly Release build directory
$candidateBuildDirs = @(
    (Join-Path $repoRoot "QtFuchsUndHenne\build\WebAssembly_Qt_6_8_3_single_threaded-Release"),
    (Join-Path $repoRoot "build\WebAssembly_Qt_6_8_3_single_threaded-Release")
)

$wasmBuild = $null
foreach ($dir in $candidateBuildDirs) {
    if (Test-Path $dir) {
        $wasmBuild = $dir
        break
    }
}

if (-not $wasmBuild) {
    Write-Error "WebAssembly build directory not found. Looked in:`n$($candidateBuildDirs -join "`n")`nPlease build the WebAssembly Release target first."
    exit 1
}

# 1. Determine remote git URL
if ([string]::IsNullOrWhiteSpace($RemoteUrl)) {
    try {
        $gitRemote = (git -C $repoRoot config --get remote.origin.url 2>$null)
        if ($gitRemote) {
            $RemoteUrl = $gitRemote.Trim()
        }
    } catch {
        # ignore git error
    }
}

if ([string]::IsNullOrWhiteSpace($RemoteUrl)) {
    $RemoteUrl = "https://github.com/mesw/FuchsUndHenne.git"
}

$stageDir = Join-Path $env:TEMP "fuchsundhenne-gh-pages"

# 2. Verify all required WebAssembly build artifacts exist
$requiredFiles = @(
    "appQtFuchsUndHenne.html",
    "appQtFuchsUndHenne.js",
    "appQtFuchsUndHenne.wasm",
    "qtloader.js"
)

# appQtFuchsUndHenne.data contains preloaded neural network weights (if embedded)
if (Test-Path (Join-Path $wasmBuild "appQtFuchsUndHenne.data")) {
    $requiredFiles += "appQtFuchsUndHenne.data"
}

foreach ($file in $requiredFiles) {
    $fullPath = Join-Path $wasmBuild $file
    if (-not (Test-Path $fullPath)) {
        Write-Error "Missing build artifact: $fullPath`nPlease rebuild the WebAssembly Release target first."
        exit 1
    }
}

Write-Host "==> Preparing clean staging directory: $stageDir" -ForegroundColor Cyan
if (Test-Path $stageDir) {
    Remove-Item -Recurse -Force $stageDir
}
New-Item -ItemType Directory -Path $stageDir | Out-Null

# 3. Copy WebAssembly build artifacts
Write-Host "==> Copying WebAssembly artifacts from: $wasmBuild" -ForegroundColor Cyan
foreach ($file in $requiredFiles) {
    Copy-Item (Join-Path $wasmBuild $file) (Join-Path $stageDir $file)
    Write-Host "    + $file" -ForegroundColor Gray
}

# Optional companion files
$optionalFiles = @("qtlogo.svg", "fox_actor.bin", "hens_actor.bin")
foreach ($opt in $optionalFiles) {
    $optPath = Join-Path $wasmBuild $opt
    if (Test-Path $optPath) {
        Copy-Item $optPath (Join-Path $stageDir $opt)
        Write-Host "    + $opt" -ForegroundColor Gray
    }
}

# 4. Copy landing page as pure UTF-8 (without BOM) and ensure link points to appQtFuchsUndHenne.html
$indexSourcePath = Join-Path $repoRoot "index.html"
$indexDestPath   = Join-Path $stageDir "index.html"
$utf8NoBom       = New-Object System.Text.UTF8Encoding($false)

if (Test-Path $indexSourcePath) {
    $htmlContent = [System.IO.File]::ReadAllText($indexSourcePath, $utf8NoBom)
    $htmlContent = $htmlContent.Replace(
        "QtFuchsUndHenne/build/WebAssembly_Qt_6_8_3_single_threaded-Release/appQtFuchsUndHenne.html",
        "appQtFuchsUndHenne.html"
    )
    [System.IO.File]::WriteAllText($indexDestPath, $htmlContent, $utf8NoBom)
    Write-Host "    + index.html (landing & redirect page)" -ForegroundColor Gray
} else {
    # Generate minimal redirect index if not found
    $fallbackHtml = @"
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta http-equiv="refresh" content="0; url=appQtFuchsUndHenne.html">
    <title>Redirecting to Fuchs und Henne...</title>
</head>
<body>
    <p>Redirecting to <a href="appQtFuchsUndHenne.html">Fuchs und Henne</a>...</p>
</body>
</html>
"@
    [System.IO.File]::WriteAllText($indexDestPath, $fallbackHtml, $utf8NoBom)
}

# 5. Create .nojekyll so GitHub Pages does not ignore underscore files or binary .wasm/.data
New-Item -ItemType File -Path (Join-Path $stageDir ".nojekyll") | Out-Null
Write-Host "    + .nojekyll" -ForegroundColor Gray

# 6. Commit and force-push ONLY the staging folder to the gh-pages branch
Write-Host "==> Deploying to $RemoteUrl (branch: $Branch)..." -ForegroundColor Cyan
Push-Location $stageDir
try {
    git init -q
    git checkout -b $Branch -q
    git add .
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    git commit -m "Deploy WebAssembly build ($timestamp)" -q
    git push -f $RemoteUrl $Branch
    
    # Extract username & repo for GitHub Pages URL display
    $pagesUrl = "https://mesw.github.io/FuchsUndHenne/"
    if ($RemoteUrl -match "github\.com[/:]([^/]+)/([^/.]+)") {
        $pagesUrl = "https://$($Matches[1]).github.io/$($Matches[2])/"
    }
    Write-Host "`n==> Successfully published to $pagesUrl" -ForegroundColor Green
} finally {
    Pop-Location
}
