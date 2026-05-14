import argparse
import importlib

try:
    import sr_corrnet  # noqa: F401
except ImportError as e:
    import sys
    sys.exit(
        f"Error: {e}\n"
        "Install with: uv sync --extra <cpu|cu126>  (or: pip install -e .)"
    )

# Parse args
parser = argparse.ArgumentParser(
    description="SR_CorrNet SS CLI — training, testing, and inference for the SS variant")
parser.add_argument(
    "--model",
    type=str,
    default="SR_CorrNet_SS",
    choices=["SR_CorrNet_SS"],
    dest="model",
    help="Model variant (only SR_CorrNet_SS is available in this fork)")
parser.add_argument(
    "--engine_mode",
    choices=["train", "inference", "test"],
    default="train",
    help="Execution mode: train, test, or inference")
parser.add_argument(
    "--config",
    type=str,
    default="baseline.yaml",
    help="Config file name (with .yaml extension)"
)
parser.add_argument(
    "--input",
    type=str,
    default=None,
    help="Input audio file or directory for inference"
)
parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Output directory for inference results"
)
parser.add_argument(
    "--gpuid",
    type=str,
    default="0",
    help="GPU device id(s), e.g. '0' or '0,1'"
)
args = parser.parse_args()

# Call target model
def resolve_module_path(model_name, module_name):
    """Determine module path within sr_corrnet package."""
    return f"sr_corrnet.models.{model_name}.{module_name}"

if args.engine_mode == "inference":
    try:
        infer_module = importlib.import_module(resolve_module_path(args.model, "main_infer"))
    except ModuleNotFoundError:
        # main_infer.py itself doesn't exist → fall back to main.py
        main_module = importlib.import_module(resolve_module_path(args.model, "main"))
        main_module.main(args)
    else:
        infer_module.main_infer(args)
else:
    main_module = importlib.import_module(resolve_module_path(args.model, "main"))
    main_module.main(args)