#!/usr/bin/env python3
"""
DartQuant v2 - Unified One-Click Quantization Script.

Usage examples:

  # INT4 with Whip loss (original DartQuant behaviour)
  python dartquant_v2/run_quantize.py \
      --model meta-llama/Llama-3.2-1B \
      --loss whip --quantizer_type int4 \
      --w_bits 4 --a_bits 4

  # INT4 with SWD-Uniform loss
  python dartquant_v2/run_quantize.py \
      --model meta-llama/Llama-3.2-1B \
      --loss swd_unif --quantizer_type int4

  # NF4 with Gaussian SWD loss
  python dartquant_v2/run_quantize.py \
      --model meta-llama/Llama-3.2-1B \
      --loss swd_gauss --quantizer_type nf4

  # With learnable Butterfly for R3/R4
  python dartquant_v2/run_quantize.py \
      --model meta-llama/Llama-3.2-1B \
      --loss swd_unif --quantizer_type int4 --butterfly
"""

import sys
import os
import logging

# Ensure project root and DartQuant paths are importable
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, 'DartQuant', 'fake_quant'))
sys.path.insert(0, os.path.join(_project_root, 'DartQuant', 'calibrater'))

from dartquant_v2.args import create_parser
from dartquant_v2.pipeline import run_full_pipeline


# Recommended loss/quantizer pairings
_RECOMMENDED_PAIRINGS = {
    'int4': ['whip', 'swd_unif'],
    'nf4': ['swd_gauss'],
}


def _validate_and_warn(args):
    """Print warnings for unusual argument combinations."""
    # Loss / quantizer pairing
    recommended = _RECOMMENDED_PAIRINGS.get(args.quantizer_type, [])
    if args.loss not in recommended:
        logging.warning(
            f"Using --loss {args.loss} with --quantizer_type {args.quantizer_type}. "
            f"Recommended pairings for {args.quantizer_type}: {recommended}. "
            f"Proceeding anyway (this is allowed for experimentation)."
        )

    # Butterfly only affects R3/R4
    if args.butterfly:
        logging.info(
            "Butterfly mode enabled: R3 and R4 will use learnable Butterfly "
            "Givens rotations instead of fixed Hadamard. R1/R2 are unaffected."
        )

    # NF4 is weight-only
    if args.quantizer_type == 'nf4' and args.a_bits < 16:
        logging.warning(
            f"NF4 is weight-only quantization but --a_bits={args.a_bits}. "
            f"Activation quantization settings will still be applied via "
            f"ActQuantWrapper for simulation purposes."
        )


def main():
    parser = create_parser()
    args = parser.parse_args()

    # Set up logging
    log_level = logging.DEBUG if hasattr(args, 'verbose') and args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    logging.info("=" * 60)
    logging.info("DartQuant v2: Unified Quantization Pipeline")
    logging.info("=" * 60)
    logging.info(f"Model:          {args.model}")
    logging.info(f"Loss:           {args.loss}")
    logging.info(f"Quantizer:      {args.quantizer_type}")
    logging.info(f"Butterfly R3/4: {args.butterfly}")
    logging.info(f"W bits:         {args.w_bits}")
    logging.info(f"A bits:         {args.a_bits}")
    logging.info(f"K bits:         {args.k_bits}")
    logging.info(f"V bits:         {args.v_bits}")
    logging.info("=" * 60)

    _validate_and_warn(args)
    run_full_pipeline(args)


if __name__ == '__main__':
    main()
