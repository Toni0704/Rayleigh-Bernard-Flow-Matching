#!/usr/bin/env python3
"""Find WHERE the 70% momentum residual lives: is it the drifted entries,
a Pr/Ra region, or spread everywhere? Run on the real bank."""
import argparse, math, torch
from verify_rb3d_multisolution import (_k, ddx_spec, ddy_spec, ddz, lap, _rms_i,
                                        ddx_fd, ddy_fd, leray_residual, _cos_basis)

p = argparse.ArgumentParser()
p.add_argument('--bank', required=True)
args = p.parse_args()
bank = torch.load(args.bank, map_location='cpu', weights_only=False)
g = bank['grid']; Gx, Gy = bank['aspect']; Nz = g['Nz']
z = g['z'].double(); dz = float(z[1]-z[0]); dx = Gx/g['Nx']; dy = Gy/g['Ny']
kx = _k(g['Nx'], Gx, 'cpu', torch.float64); ky = _k(g['Ny'], Gy, 'cpu', torch.float64)
cos = _cos_basis(Nz, 'cpu', torch.float64)

rows = []
for (Ra, Pr, mode), e in bank['entries'].items():
    if not e['converged']:
        continue
    u = e['grid_u'].double(); v = e['grid_v'].double(); w = e['grid_w'].double()
    T = e['grid_T'].double(); Tp = T - (1.0-z)[None,None,:]
    def advF(a): return u*ddx_fd(a,dx)+v*ddy_fd(a,dy)+w*ddz(a,dz)
    Ru = advF(u) - Pr*lap(u,kx,ky,dz)
    Rv = advF(v) - Pr*lap(v,kx,ky,dz)
    Rw = advF(w) - Pr*lap(w,kx,ky,dz) - Pr*Ra*Tp
    Ru,Rv,Rw = leray_residual(Ru,Rv,Rw,kx,ky,dz,cos)
    sM = max(_rms_i(Pr*Ra*Tp),1e-12)
    rmom = math.sqrt(_rms_i(Ru)**2+_rms_i(Rv)**2+_rms_i(Rw)**2)/sM
    drift = tuple(e['planform']) != tuple(mode)
    rows.append((rmom, Ra, Pr, mode, tuple(e['planform']), drift))

rows.sort(reverse=True)
print(f"total converged: {len(rows)}")
print(f"\n=== 15 WORST momentum residuals ===")
for rmom,Ra,Pr,mode,pf,drift in rows[:15]:
    print(f"  mom={rmom:.3f} Ra={Ra:6.0f} Pr={Pr:4.2f} seed={mode} -> {pf} {'DRIFT' if drift else ''}")

drift_r = [r[0] for r in rows if r[5]]
clean_r = [r[0] for r in rows if not r[5]]
def stats(x): 
    if not x: return "none"
    t=torch.tensor(x); return f"n={len(x)} median={t.median():.3f} p90={t.quantile(0.9):.3f} max={t.max():.3f}"
print(f"\n=== momentum residual by category ===")
print(f"  DRIFTED entries: {stats(drift_r)}")
print(f"  CLEAN entries  : {stats(clean_r)}")

# how many clean entries would pass at various thresholds?
import numpy as np
ct = torch.tensor(clean_r)
for thr in (0.10, 0.15, 0.20, 0.30):
    print(f"  clean entries with mom < {thr}: {int((ct<thr).sum())}/{len(clean_r)} ({100*float((ct<thr).float().mean()):.1f}%)")
