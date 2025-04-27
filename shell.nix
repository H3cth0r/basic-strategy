{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    pkgs.python3
    pkgs.python3Packages.virtualenv
    
    # System dependencies
    pkgs.libffi
    pkgs.zlib
    pkgs.stdenv.cc.cc.lib  # Essential for PyTorch/C++ extensions
    
    # Uncomment for CUDA support (if needed)
    # pkgs.cudatoolkit
    # pkgs.cudnn
  ];

  shellHook = ''
    # Create virtual environment if missing
    if [ ! -d "venv" ]; then
      virtualenv venv
    fi

    # Activate virtual environment
    source venv/bin/activate

    # Install requirements
    if [ -f "requirements.txt" ]; then
      # Ensure pip is upgraded first
      pip install --upgrade pip
      pip install -r requirements.txt
    else
      echo "No requirements.txt found"
    fi

    # Help Python find libstdc++
    export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
  '';
}
