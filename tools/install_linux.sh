#!/usr/bin/env bash
# Installs AstroSharp for the current user on Linux, with no root needed (the Linux
# counterpart of `make install` on macOS and the Windows installer):
#   - astrosharp-gui and astrosharp-cli in ~/.local/bin;
#   - a desktop launcher with the app icon in ~/.local/share.
#
#   tools/install_linux.sh               build (release) and install
#   tools/install_linux.sh --uninstall   remove what it installed
#
# BINDIR (default ~/.local/bin) and XDG_DATA_HOME (default ~/.local/share) change where the
# files go.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$(uname -s)" != Linux ]; then
    echo "install_linux.sh is for Linux; on macOS use make install" >&2
    exit 1
fi

bindir=${BINDIR:-$HOME/.local/bin}
datadir=${XDG_DATA_HOME:-$HOME/.local/share}
# Must match the GUI's app ID (crates/gui/src/main.rs), so the desktop matches its windows
# to this launcher and shows the right icon.
app_id=astrosharp
desktop=$datadir/applications/$app_id.desktop
icons=$datadir/icons/hicolor
installed=(
    "$bindir/astrosharp-gui"
    "$bindir/astrosharp-cli"
    "$desktop"
    "$icons/scalable/apps/$app_id.svg"
    "$icons/512x512/apps/$app_id.png"
)

# Lets the desktop pick up added or removed launchers and icons straight away.
refresh() {
    if command -v update-desktop-database >/dev/null; then
        update-desktop-database -q "$datadir/applications" || true
    fi
    if command -v gtk-update-icon-cache >/dev/null && [ -d "$icons" ]; then
        gtk-update-icon-cache -q -t -f "$icons" 2>/dev/null || true
    fi
}

if [ "${1:-}" = --uninstall ]; then
    rm -f "${installed[@]}"
    refresh
    echo "Removed AstroSharp from $bindir and $datadir"
    exit 0
fi

cargo build --release -p astrosharp-gui -p astrosharp-cli
version=$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)

install -Dm755 target/release/astrosharp-gui "$bindir/astrosharp-gui"
# The same name as in the macOS bundle and the Windows installer, so scripts (such as
# Siril's) work everywhere.
install -Dm755 target/release/astrosharp "$bindir/astrosharp-cli"
install -Dm644 assets/icon.svg "$icons/scalable/apps/$app_id.svg"
install -Dm644 assets/icon-512.png "$icons/512x512/apps/$app_id.png"
mkdir -p "$(dirname "$desktop")"
cat > "$desktop" <<EOF
[Desktop Entry]
Type=Application
Name=AstroSharp
GenericName=Astronomical Image Sharpener
Comment=Sharpen and denoise stretched astronomical images
Exec="$bindir/astrosharp-gui" %f
Icon=$app_id
Terminal=false
Categories=Science;Astronomy;
Keywords=astrophotography;sharpen;denoise;FITS;TIFF;
StartupWMClass=$app_id
EOF
if command -v desktop-file-validate >/dev/null; then
    desktop-file-validate "$desktop"
fi
refresh

echo "Installed AstroSharp $version: $bindir/astrosharp-gui, $bindir/astrosharp-cli and $desktop"
case ":$PATH:" in
    *":$bindir:"*) ;;
    *) echo "note: $bindir isn't on your PATH yet; most distributions add it at the next login" ;;
esac
