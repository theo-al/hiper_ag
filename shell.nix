{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  packages = with pkgs; [
    uv
    python311Packages.numpy
    python311Packages.matplotlib

    pandoc
    texlive.combined.scheme-small
  ];
  shellHook = ''
    set +ex
    if [ ! -d ".venv" ]; then
      uv venv --system-site-packages
      uv pip install geneticalgorithm2
      uv pip uninstall numpy
    fi
    source .venv/bin/activate
  '';
}

