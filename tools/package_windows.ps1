# Builds AstroSharp for Windows on ARM64 and x64 and packages both as one Inno Setup
# installer in target\release\bundle\windows\ (the Windows counterpart of
# tools/bundle_macos.sh). The installer puts the native build on each machine.
#
#   powershell -ExecutionPolicy Bypass -File tools\package_windows.ps1
#
# Runs on Windows on ARM. Needs:
#   - the native ARM64 Rust toolchain, with the x64 target added (it needn't be the default):
#       rustup toolchain install stable-aarch64-pc-windows-msvc
#       rustup target add x86_64-pc-windows-msvc --toolchain stable-aarch64-pc-windows-msvc
#   - Visual Studio 2022 Build Tools with "MSVC v143 - VS 2022 C++ ARM64/ARM64EC build tools",
#     "MSVC v143 - VS 2022 C++ x64/x86 build tools" and the Windows SDK;
#   - Inno Setup 6 (winget install --id JRSoftware.InnoSetup -e).

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')

# Native ARM64 compiler even when the default toolchain is x64 (which would run emulated);
# it cross-compiles the x64 build.
$toolchain = 'stable-aarch64-pc-windows-msvc'
if (-not (rustup toolchain list | Select-String -SimpleMatch $toolchain)) {
    throw "The native ARM64 Rust toolchain is missing; install it with: rustup toolchain install $toolchain"
}
# Installer folder name for each Rust target.
$targets = [ordered]@{ 'aarch64-pc-windows-msvc' = 'arm64'; 'x86_64-pc-windows-msvc' = 'x64' }
$installed = rustup target list --installed --toolchain $toolchain
foreach ($target in $targets.Keys) {
    if ($installed -notcontains $target) {
        throw "The $target Rust target is missing; add it with: rustup target add $target --toolchain $toolchain"
    }
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

$version = ((cargo metadata --no-deps --format-version 1 | ConvertFrom-Json).packages |
    Where-Object name -eq 'astrosharp-gui').version

$out = 'target\release\bundle\windows'
$stage = "$out\stage"
Remove-Item -Recurse -Force $stage -ErrorAction SilentlyContinue
foreach ($target in $targets.Keys) {
    cargo "+$toolchain" build --release --target $target -p astrosharp-gui -p astrosharp-cli
    if ($LASTEXITCODE -ne 0) { throw "cargo build for $target failed" }
    $dir = "$stage\$($targets[$target])"
    New-Item -ItemType Directory -Force $dir | Out-Null
    # The CLI gets a distinct name: Windows file names are case-insensitive, so astrosharp.exe
    # would clash with AstroSharp.exe.
    Copy-Item "target\$target\release\astrosharp-gui.exe" "$dir\AstroSharp.exe"
    Copy-Item "target\$target\release\astrosharp.exe" "$dir\astrosharp-cli.exe"
}

& $iscc /Q "/DAppVersion=$version" "/DSourceDir=$((Resolve-Path $stage).Path)" `
    "/O$((Resolve-Path $out).Path)" tools\windows\astrosharp.iss
if ($LASTEXITCODE -ne 0) { throw 'Inno Setup failed' }
Write-Host "Built $out\AstroSharp-$version-setup.exe"
