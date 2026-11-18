"""Command-line entry point for the PathSnap development workflow."""

import argparse

from map_matching import candidates, demo, feature_export, inventory


def main() -> None:
    parser = argparse.ArgumentParser(prog="python main.py", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, module in (
        ("inspect", inventory),
        ("candidates", candidates),
        ("features", feature_export),
        ("demo", demo),
    ):
        command = commands.add_parser(name, help=module.__doc__, description=module.__doc__)
        module.add_arguments(command)
        command.set_defaults(handler=module.run_command, command_parser=command)

    args = parser.parse_args()
    try:
        args.handler(args)
    except (OSError, ValueError) as exc:
        args.command_parser.error(str(exc))


if __name__ == "__main__":
    main()
