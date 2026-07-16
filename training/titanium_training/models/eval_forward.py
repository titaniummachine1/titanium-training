"""Shared raw HalfPW forward pass — must match ``search.rs::evaluate()`` (normed=False).

Used by ``halfpw.py``, ``trainer.HalfPW``, and parity tests so training optimizes the
same function the engine executes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from titanium_training.models.field_planes import (
    CAT_RAW_ME,
    CAT_RAW_OPP,
    CAT_PROPAGATED_ME,
    CAT_PROPAGATED_OPP,
    CAT_PROPAGATED_COMBINED,
    ROUTE_CONTESTED,
    ROUTE_ME,
    ROUTE_NEAR_ME,
    ROUTE_NEAR_OPP,
    ROUTE_OPP,
    compact_catv5_precise_vectors,
    compact_route_vectors,
)

NET_H = 32
NET_MIRC = [(8 - i // 9) * 9 + (8 - i % 9) for i in range(81)]
NET_MIRS = [(7 - i // 8) * 8 + (7 - i % 8) for i in range(64)]
NET_BKT = [(i // 9 // 3) * 3 + (i % 9) // 3 for i in range(81)]


@dataclass
class EvalTrace:
  """Intermediate tensors for tri-path parity (trainer / halfpw / Rust)."""

  scalar_inputs: dict[str, float]
  scalar_out: float
  route_out: float
  cat_out: float
  width_contrib: float
  wall_acc: list[float]
  hidden_pre: list[float]
  hidden_clip: list[float]
  neural_out: float
  final_cp: int


def opponent_corridor_width(rec: dict, me: int, d_opp_i: int) -> float:
  """ws[15] from eval JSON — prefer engine ``corridor_width*`` when present."""
  if me == 0:
    key = "corridor_width1"
  else:
    key = "corridor_width0"
  if key in rec:
    return float(rec[key])
  from titanium_training.models.halfpw import opponent_corridor_width as _legacy

  return float(_legacy(rec, me, 0, d_opp_i))


def raw_scalar_block(
  ws,
  d_me_raw: float,
  d_opp_raw: float,
  w_me_raw: float,
  w_opp_raw: float,
) -> tuple[dict[str, float], float]:
  """Scalar skip through ws[13] — engine order before route / width / hidden."""
  d_me = float(d_me_raw)
  d_opp = float(d_opp_raw)
  w_me = float(w_me_raw)
  w_opp = float(w_opp_raw)
  pd = d_opp - d_me
  wd = w_me - w_opp
  inputs = {
    "d_me": d_me,
    "d_opp": d_opp,
    "w_me": w_me,
    "w_opp": w_opp,
    "pd": pd,
    "wd": wd,
  }
  out = (
    ws[0]
    + ws[1] * pd
    + ws[2] * wd
    + ws[3] * d_me
    + ws[4] * d_opp
    + ws[9] * pd * (w_me + w_opp) / 20.0
    + ws[10] * wd * (d_me + d_opp) / 16.0
  )
  if w_opp == 0.0:
    out += ws[6]
    if d_me <= d_opp:
      out += ws[5]
  elif w_me == 0.0:
    out += ws[8]
    if d_opp <= d_me - 1.0:
      out += ws[7]
  if d_opp <= 4.0:
    out += ws[11] * (w_me if w_me < 3.0 else 3.0)
  if d_me <= 4.0:
    out += ws[12] * (w_opp if w_opp < 3.0 else 3.0)
  out += ws[13] * pd * w_opp / 10.0
  return inputs, float(out)


def wall_accumulator(w1c, bucket: int, hw, vw, me: int, pawn0: int, pawn1: int, h: int = NET_H) -> list[float]:
  """Sum ``w1c`` for placed walls — matches ``search.rs`` / legacy ``halfpw.py``."""
  acc = [0.0] * h
  if me == 0:
    for s in range(64):
      if hw[s]:
        o = (bucket * 128 + s) * h
        for j in range(h):
          acc[j] += w1c[o + j]
      if vw[s]:
        o = (bucket * 128 + 64 + s) * h
        for j in range(h):
          acc[j] += w1c[o + j]
  else:
    for s in range(64):
      if hw[s]:
        o = (bucket * 128 + NET_MIRS[s]) * h
        for j in range(h):
          acc[j] += w1c[o + j]
      if vw[s]:
        o = (bucket * 128 + 64 + NET_MIRS[s]) * h
        for j in range(h):
          acc[j] += w1c[o + j]
  return acc


def route_and_cat_out(net, rec: dict, me: int) -> tuple[float, float]:
  route_me, route_opp, near_me, near_opp, contested = compact_route_vectors(rec, NET_MIRC)
  cat_raw_me, cat_raw_opp, cat_prop_me, cat_prop_opp, cat_combined = compact_catv5_precise_vectors(rec, NET_MIRC)
  route_out = sum(
    net.route_me[i] * route_me[i]
    + net.route_opp[i] * route_opp[i]
    + net.route_near_me[i] * near_me[i]
    + net.route_near_opp[i] * near_opp[i]
    + net.route_contested[i] * contested[i]
    for i in range(81)
  )
  cat_out = sum(
    net.cat_raw_me[i] * cat_raw_me[i]
    + net.cat_raw_opp[i] * cat_raw_opp[i]
    + net.cat_propagated_me[i] * cat_prop_me[i]
    + net.cat_propagated_opp[i] * cat_prop_opp[i]
    + net.cat_propagated_combined[i] * cat_combined[i]
    for i in range(81)
  )
  return float(route_out), float(cat_out)


def hidden_block(net, acc: list[float], me: int, pawn0: int, pawn1: int) -> tuple[list[float], list[float], float]:
  h = net.h
  hidden_pre = [0.0] * h
  if me == 0:
    po_base = pawn0 * h
    px_base = pawn1 * h
  else:
    po_base = NET_MIRC[pawn1] * h
    px_base = NET_MIRC[pawn0] * h
  for j in range(h):
    hidden_pre[j] = net.b1[j] + acc[j] + net.po[po_base + j] + net.px[px_base + j]
  hidden_clip = [min(1.0, max(0.0, v)) for v in hidden_pre]
  neural_out = sum(net.w2[j] * hidden_clip[j] * 200.0 for j in range(h))
  return hidden_pre, hidden_clip, float(neural_out)


def forward_trace_from_record(net, rec: dict) -> EvalTrace:
  me = int(rec["turn"])
  wl = [rec["wl0"], rec["wl1"]]
  dist = [rec["d0"], rec["d1"]]
  d_me_raw = float(dist[me])
  d_opp_raw = float(dist[1 - me])
  w_me_raw = float(wl[me])
  w_opp_raw = float(wl[1 - me])
  ws = net.ws
  pawn0, pawn1 = int(rec["pawn0"]), int(rec["pawn1"])
  hw = list(rec["hw"])
  vw = list(rec["vw"])

  scalar_inputs, scalar_out = raw_scalar_block(ws, d_me_raw, d_opp_raw, w_me_raw, w_opp_raw)
  route_out, cat_out = route_and_cat_out(net, rec, me)
  d_opp_i = int(d_opp_raw)
  width_opp = opponent_corridor_width(rec, me, d_opp_i)
  scalar_inputs["width_opp"] = width_opp
  width_contrib = float(ws[15] * width_opp)

  if me == 0:
    bucket = NET_BKT[pawn0]
  else:
    bucket = NET_BKT[NET_MIRC[pawn1]]

  acc = wall_accumulator(net.w1c, bucket, hw, vw, me, pawn0, pawn1, h=net.h)
  hidden_pre, hidden_clip, neural_out = hidden_block(net, acc, me, pawn0, pawn1)

  total = scalar_out + route_out + cat_out + width_contrib + neural_out
  return EvalTrace(
    scalar_inputs=scalar_inputs,
    scalar_out=scalar_out,
    route_out=route_out,
    cat_out=cat_out,
    width_contrib=width_contrib,
    wall_acc=acc,
    hidden_pre=hidden_pre,
    hidden_clip=hidden_clip,
    neural_out=neural_out,
    final_cp=int(total),
  )


def record_to_trainer_batch(rec: dict) -> dict[str, Any]:
  """Build a batch-of-1 dict matching ``QuoridorDataset.__getitem__``."""
  import torch

  me = int(rec["turn"])
  d_me = rec["d0"] if me == 0 else rec["d1"]
  d_opp = rec["d1"] if me == 0 else rec["d0"]
  w_me = rec["wl0"] if me == 0 else rec["wl1"]
  w_opp = rec["wl1"] if me == 0 else rec["wl0"]
  width_opp = opponent_corridor_width(rec, me, int(d_opp))
  route_me, route_opp, route_near_me, route_near_opp, route_contested = compact_route_vectors(
    rec, NET_MIRC
  )
  cat_raw_me, cat_raw_opp, cat_prop_me, cat_prop_opp, cat_combined = compact_catv5_precise_vectors(rec, NET_MIRC)
  hw = list(rec["hw"])
  vw = list(rec["vw"])
  if me == 0:
    pawn_me_idx = int(rec["pawn0"])
    pawn_opp_idx = int(rec["pawn1"])
    bucket = NET_BKT[pawn_me_idx]
    wall_mask = hw + vw
  else:
    pawn_me_idx = NET_MIRC[int(rec["pawn1"])]
    pawn_opp_idx = NET_MIRC[int(rec["pawn0"])]
    bucket = NET_BKT[pawn_me_idx]
    wall_mask = [hw[NET_MIRS[s]] for s in range(64)] + [vw[NET_MIRS[s]] for s in range(64)]

  return {
    "d_me": torch.tensor([d_me], dtype=torch.float32),
    "d_opp": torch.tensor([d_opp], dtype=torch.float32),
    "w_me": torch.tensor([w_me], dtype=torch.float32),
    "w_opp": torch.tensor([w_opp], dtype=torch.float32),
    "width_opp": torch.tensor([width_opp], dtype=torch.float32),
    ROUTE_ME: torch.tensor([route_me], dtype=torch.float32),
    ROUTE_OPP: torch.tensor([route_opp], dtype=torch.float32),
    ROUTE_NEAR_ME: torch.tensor([route_near_me], dtype=torch.float32),
    ROUTE_NEAR_OPP: torch.tensor([route_near_opp], dtype=torch.float32),
    ROUTE_CONTESTED: torch.tensor([route_contested], dtype=torch.float32),
    CAT_RAW_ME: torch.tensor([cat_raw_me], dtype=torch.float32),
    CAT_RAW_OPP: torch.tensor([cat_raw_opp], dtype=torch.float32),
    CAT_PROPAGATED_ME: torch.tensor([cat_prop_me], dtype=torch.float32),
    CAT_PROPAGATED_OPP: torch.tensor([cat_prop_opp], dtype=torch.float32),
    CAT_PROPAGATED_COMBINED: torch.tensor([cat_combined], dtype=torch.float32),
    "bucket": torch.tensor([bucket], dtype=torch.long),
    "wall_mask": torch.tensor([wall_mask], dtype=torch.float32),
    "pawn_me": torch.tensor([pawn_me_idx], dtype=torch.long),
    "pawn_opp": torch.tensor([pawn_opp_idx], dtype=torch.long),
  }
