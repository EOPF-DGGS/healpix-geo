#!/usr/bin/env python3

import pathlib


def main():
    root = pathlib.Path(__file__).parent

    local_channel = root.parent / "target/conda"
    local_packages = list(local_channel.glob("emscripten-wasm32/healpix-geo*"))
    if len(local_packages) != 1:
        raise RuntimeError(f"zero or more than one package found: {local_packages}")

    local_package = local_packages[0]

    template_path = root / "environment_template.yml"
    env_path = root / "environment.yml"

    template = template_path.read_text()
    environment = template.replace('"{{ local-package }}"', local_package)

    env_path.write_text(environment)


if __name__ == "__main__":
    main()
