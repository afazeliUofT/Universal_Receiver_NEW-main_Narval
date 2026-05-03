from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from upair5g.builders import build_pusch_transmitter, extract_pilot_mask_per_stream, get_resource_grid  # noqa: E402
from upair5g.config import get_cfg, load_config, set_cfg  # noqa: E402
from upair5g.utils import call_transmitter  # noqa: E402


DMRS_CASES = {
    "1dmrs": {
        "multiuser.dmrs.length": 1,
        "multiuser.dmrs.additional_position": 0,
    },
    "2dmrs": {
        "multiuser.dmrs.length": 1,
        "multiuser.dmrs.additional_position": 1,
    },
}


def _streams_from_x(x: np.ndarray, num_users: int) -> np.ndarray:
    # Expected Sionna shape: [B, num_tx, num_tx_ant, T, F].
    if x.ndim != 5:
        raise ValueError(f"Expected transmitter grid rank 5, got shape {x.shape}")
    if x.shape[1] < num_users:
        raise ValueError(f"Transmitter returned only {x.shape[1]} users, expected {num_users}")
    return x[0, :num_users, 0, :, :]


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe multi-user PUSCH DMRS orthogonality for the installed Sionna version.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "twc_comprehensive_mu32_base.yaml"))
    parser.add_argument("--dmrs-case", choices=sorted(DMRS_CASES), default=None)
    parser.add_argument("--num-users", type=int, default=4)
    parser.add_argument("--tol", type=float, default=1e-5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.dmrs_case is not None:
        for key, value in DMRS_CASES[args.dmrs_case].items():
            set_cfg(cfg, key, value)
    tx, _ = build_pusch_transmitter(cfg, num_users=args.num_users)
    rg = get_resource_grid(tx)
    mask = np.asarray(extract_pilot_mask_per_stream(rg).numpy(), dtype=np.float32)
    if mask.shape[-1] < args.num_users:
        raise RuntimeError(f"Pilot mask exposes {mask.shape[-1]} streams, expected {args.num_users}.")
    mask = mask[..., : args.num_users]
    union_mask = np.max(mask, axis=-1) > 0.5

    x, _ = call_transmitter(tx, batch_size=1)
    streams = _streams_from_x(np.asarray(x.numpy()), args.num_users)
    pilots = streams[:, union_mask]
    gram = pilots @ np.conjugate(pilots.T)
    norm = np.sqrt(np.maximum(np.real(np.diag(gram)), 1e-12))
    gram_norm = gram / np.maximum(norm[:, None] * norm[None, :], 1e-12)

    print("Sionna multi-user DMRS probe")
    print(f"config: {Path(args.config).resolve()}")
    print(f"dmrs_case: {args.dmrs_case or 'config-default'}")
    print(f"num_users: {args.num_users}")
    print(f"configured port_sets: {get_cfg(cfg, 'multiuser.dmrs.port_sets', None)}")
    print(f"pilot mask shape [T,F,U]: {mask.shape}")
    print(f"DMRS REs per user: {[int(mask[..., i].sum()) for i in range(args.num_users)]}")
    print("Pairwise pilot-mask overlaps:")
    print(np.asarray([[int(np.sum((mask[..., i] > 0.5) & (mask[..., j] > 0.5))) for j in range(args.num_users)] for i in range(args.num_users)]))
    print("Normalized |DMRS Gram| over union DMRS REs:")
    print(np.abs(gram_norm))

    offdiag = np.abs(gram_norm - np.diag(np.diag(gram_norm)))
    max_offdiag = float(np.max(offdiag))
    print(f"max off-diagonal normalized inner product: {max_offdiag:.3e}")
    if max_offdiag > float(args.tol):
        raise SystemExit(f"FAILED: DMRS are not orthogonal under tol={args.tol}")
    print("PASSED: DMRS streams are orthogonal under the requested tolerance.")


if __name__ == "__main__":
    main()
