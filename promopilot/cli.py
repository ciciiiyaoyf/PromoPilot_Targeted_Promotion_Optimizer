"""Command-line entry points with explicit input/output locations."""
import argparse
from pathlib import Path

from .config import PipelineConfig
from .data import DataQualityError, load_tables, validate_and_clean


def main(argv=None):
    parser = argparse.ArgumentParser(prog='promopilot', description='Validate, model, and allocate customer promotions.')
    commands = parser.add_subparsers(dest='command', required=True)
    validate = commands.add_parser('validate', help='Check the eleven raw CSV extracts without fitting models')
    validate.add_argument('--data-dir', required=True, type=Path)
    run = commands.add_parser('run', help='Run the complete evaluated modeling and MILP pipeline')
    run.add_argument('--data-dir', required=True, type=Path)
    run.add_argument('--output-dir', default=Path('artifacts'), type=Path)
    run.add_argument('--bootstrap-draws', default=2000, type=int)
    args = parser.parse_args(argv)
    try:
        if args.command == 'validate':
            _, report = validate_and_clean(load_tables(args.data_dir))
            print(report.to_string(index=False))
            print(f'Validation passed: {len(report)} checks; {report.status.eq("WARN").sum()} warnings.')
        else:
            from .pipeline import run as run_pipeline
            run_pipeline(args.data_dir, args.output_dir, PipelineConfig(bootstrap_draws=args.bootstrap_draws))
    except (DataQualityError, ValueError, FileNotFoundError) as error:
        parser.exit(2, f'{error}\n')
    return 0
