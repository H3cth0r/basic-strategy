{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    pkgs.python312
    pkgs.python312Packages.virtualenv
    pkgs.zlib  # Explicitly include zlib
  ];

  # Add all necessary libraries to LD_LIBRARY_PATH
  LD_LIBRARY_PATH = "${pkgs.lib.makeLibraryPath [
    pkgs.stdenv.cc.cc.lib  # C++ library
    pkgs.zlib              # zlib library
  ]}";

  shellHook = ''
    # Create and activate virtual environment if not present
    VENV_DIR="$(pwd)/.venv"
    if [ ! -d "$VENV_DIR" ]; then
      echo "Creating virtual environment..."
      python -m venv "$VENV_DIR"
      source "$VENV_DIR/bin/activate"
      pip install --upgrade pip setuptools
      pip install numpy pandas matplotlib yfinance torch seaborn plotly scikit-learn ta tqdm gym gymnasium flask
    else
      source "$VENV_DIR/bin/activate"
    fi
  '';
}
