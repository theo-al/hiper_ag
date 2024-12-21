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
    uv venv --system-site-packages
    source .venv/bin/activate
    uv pip install geneticalgorithm2
  '';
}

