#!/usr/bin/env bash
# Builds an unsigned AstroSharp.app for local use in target/release/bundle/.
set -euo pipefail
cd "$(dirname "$0")/.."

cargo build --release -p astrosharp-gui

version=$(cargo metadata --no-deps --format-version 1 |
    python3 -c 'import json,sys; print(next(p["version"] for p in json.load(sys.stdin)["packages"] if p["name"] == "astrosharp-gui"))')
app=target/release/bundle/AstroSharp.app
rm -rf "$app"
mkdir -p "$app/Contents/MacOS"
cp target/release/astrosharp-gui "$app/Contents/MacOS/AstroSharp"

cat > "$app/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>AstroSharp</string>
    <key>CFBundleDisplayName</key><string>AstroSharp</string>
    <key>CFBundleIdentifier</key><string>local.astrosharp</string>
    <key>CFBundleExecutable</key><string>AstroSharp</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>${version}</string>
    <key>CFBundleVersion</key><string>${version}</string>
    <key>LSMinimumSystemVersion</key><string>11.0</string>
    <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
EOF

echo "Built $app"
