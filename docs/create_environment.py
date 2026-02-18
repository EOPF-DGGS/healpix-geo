#!/usr/bin/env python3

import pathlib


def main():
    root = pathlib.Path(__file__).parent

    local_channel = f"file://{root.parent / 'target/conda'}"

    template_path = root / "environment_template.yml"
    env_path = root / "environment.yml"

    template = template_path.read_text()
    environment = template.replace("{{ local-channel }}", local_channel)

    env_path.write_text(environment)


if __name__ == "__main__":
    main()
