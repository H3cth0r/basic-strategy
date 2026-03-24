{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    pkgs.python3
    pkgs.python3Packages.pip
    pkgs.python3Packages.virtualenv
  ];

  shellHook = ''
    echo "Setting up Python virtual environment..."
    # Create a virtual environment directory named .nix-venv to avoid colliding with your .venv file
    if [ ! -d ".nix-venv" ]; then
      python -m venv .nix-venv
    fi
    source .nix-venv/bin/activate
    
    echo "Installing required modules via pip..."
    pip install requests pandas python-dotenv
  '';
}