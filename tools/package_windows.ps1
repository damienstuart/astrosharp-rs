# Builds AstroSharp for Windows on ARM and packages it as an Inno Setup installer in
# target\release\bundle\windows\ (the Windows counterpart of tools/bundle_macos.sh).
#
#   powershell -ExecutionPolicy Bypass -File tools\package_windows.ps1
#
# Needs the native ARM64 Rust toolchain (rustup toolchain install stable-aarch64-pc-windows-msvc;
# it needn't be the default), Visual Studio 2022 Build Tools with "MSVC v143 - VS 2022 C++
# ARM64/ARM64EC build tools" and the Windows SDK, and Inno Setup 6
# (winget install --id JRSoftware.InnoSetup -e).

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')

# Native ARM64 build even when the default toolchain is x64 (which would run emulated).
$toolchain = 'stable-aarch64-pc-windows-msvc'
if (-not (rustup toolchain list | Select-String -SimpleMatch $toolchain)) {
    throw "The native ARM64 Rust toolchain is missing; install it with: rustup toolchain install $toolchain"
}

$iscc = (Get-Command ISCC.exe -ErrorAction SilentlyContinue).Source
if (-not $iscc) {
    $iscc = @(
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
        "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
    ) | Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (-not $iscc) {
    throw 'Inno Setup 6 not found; install it with: winget install --id JRSoftware.InnoSetup -e'
}

cargo "+$toolchain" build --release -p astrosharp-gui -p astrosharp-cli
if ($LASTEXITCODE -ne 0) { throw 'cargo build failed' }

$version = ((cargo metadata --no-deps --format-version 1 | ConvertFrom-Json).packages |
    Where-Object name -eq 'astrosharp-gui').version

$out = 'target\release\bundle\windows'
$stage = "$out\stage"
Remove-Item -Recurse -Force $stage -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $stage | Out-Null
# The CLI gets a distinct name: Windows file names are case-insensitive, so astrosharp.exe
# would clash with AstroSharp.exe.
Copy-Item target\release\astrosharp-gui.exe "$stage\AstroSharp.exe"
Copy-Item target\release\astrosharp.exe "$stage\astrosharp-cli.exe"

& $iscc /Q "/DAppVersion=$version" "/DSourceDir=$((Resolve-Path $stage).Path)" `
    "/O$((Resolve-Path $out).Path)" tools\windows\astrosharp.iss
if ($LASTEXITCODE -ne 0) { throw 'Inno Setup failed' }
Write-Host "Built $out\AstroSharp-$version-arm64-setup.exe"
