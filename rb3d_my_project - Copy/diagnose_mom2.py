#!/usr/bin/env python3
"""Test the hypothesis: the 70% momentum residual is a Pr-amplified operator
mismatch in the viscous term, worst at low-Ra/high-Pr where the buoyancy
normalizer is smallest. Checks (a) residual vs Pr, (b) residual with an
ABSOLUTE normalizer instead of the tiny buoyancy term."""
import argparse, math, torch
from verify_rb3d_multisolution import (_k, ddz, lap, _rms_i, ddx_fd, ddy_fd,
                                        leray_residual, _cos_basis)
p=argparse.ArgumentParser(); p.add_argument('--bank',required=True); a=p.parse_args()
bank=torch.load(a.bank,map_location='cpu',weights_only=False)
g=bank['grid']; Gx,Gy=bank['aspect']; Nz=g['Nz']
z=g['z'].double(); dz=float(z[1]-z[0]); dx=Gx/g['Nx']; dy=Gy/g['Ny']
kx=_k(g['Nx'],Gx,'cpu',torch.float64); ky=_k(g['Ny'],Gy,'cpu',torch.float64)
cos=_cos_basis(Nz,'cpu',torch.float64)

buckets={}  # Pr bin -> list of (rel_buoy, rel_abs)
for (Ra,Pr,mode),e in bank['entries'].items():
    if not e['converged']: continue
    u=e['grid_u'].double(); v=e['grid_v'].double(); w=e['grid_w'].double()
    Tp=e['grid_T'].double()-(1.0-z)[None,None,:]
    def advF(x): return u*ddx_fd(x,dx)+v*ddy_fd(x,dy)+w*ddz(x,dz)
    Ru=advF(u)-Pr*lap(u,kx,ky,dz); Rv=advF(v)-Pr*lap(v,kx,ky,dz)
    Rw=advF(w)-Pr*lap(w,kx,ky,dz)-Pr*Ra*Tp
    Ru,Rv,Rw=leray_residual(Ru,Rv,Rw,kx,ky,dz,cos)
    resid=math.sqrt(_rms_i(Ru)**2+_rms_i(Rv)**2+_rms_i(Rw)**2)
    # normalizer A: buoyancy term (current); B: the advection+viscous SCALE
    s_buoy=max(_rms_i(Pr*Ra*Tp),1e-12)
    s_visc=max(_rms_i(Pr*lap(u,kx,ky,dz)),_rms_i(Pr*lap(w,kx,ky,dz)),1e-12)
    s_adv=max(_rms_i(advF(u)),_rms_i(advF(w)),1e-12)
    s_abs=max(s_buoy,s_visc,s_adv)   # dominant balancing term, whichever it is
    b=round(Pr); buckets.setdefault(b,[]).append((resid/s_buoy, resid/s_abs))
print(f"{'Pr~':>4} {'n':>4} {'rel/buoyancy(med)':>18} {'rel/dominant(med)':>18}")
for b in sorted(buckets):
    t=torch.tensor(buckets[b])
    print(f"{b:>4} {len(t):>4} {float(t[:,0].median()):>18.3f} {float(t[:,1].median()):>18.3f}")
allt=torch.tensor([x for v in buckets.values() for x in v])
print(f"\nOVERALL max rel/buoyancy = {float(allt[:,0].max()):.3f}")
print(f"OVERALL max rel/dominant  = {float(allt[:,1].max()):.3f}")
print(f"clean-ish frac rel/dominant<0.1: {100*float((allt[:,1]<0.1).float().mean()):.1f}%")
