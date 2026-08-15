"""
Class 4: Relativistic Mean-Field (RMF) Models
"""
from enum import Enum, auto
import numpy as np
from scipy.optimize import brentq, root
from eos_common import (
    n0, mN, NB_GRID, N_GRID, cs2_acceptance_check,
    hc, hc3,
    fermi_energy_integral as _fe,
    fermi_pressure_integral as _fp,
    fermi_scalar_integral as _fs,
    kF_from_density as _kF,
    density_from_kF as _n_from_kF,
)
from sampling_utils import uniform, normal_truncated, beta_scaled

CLASS_NAME = "Class 4: Relativistic Mean-Field (RMF)"
FILE_PREFIX = "class4_rmf"

m_e  = 0.511
m_mu = 105.658
M_OMEGA = 783.0
M_RHO   = 763.0
E_SAT = -16.0

K0_MIN, K0_MAX = 200.0, 300.0
MSTAR_RATIO_MIN, MSTAR_RATIO_MAX = 0.55, 0.80
J_MIN, J_MAX = 28.0, 36.0
L_MIN, L_MAX = 30.0, 90.0
MSIGMA_MIN, MSIGMA_MAX = 400.0, 600.0
ZETA_MIN, ZETA_MAX = 0.0, 0.06
LAMBDA_W_MIN, LAMBDA_W_MAX = 0.0, 0.05
DD_VARIATION = 0.30

class Rejection(Enum):
    COUPLING_FAILED  = auto()
    L_OUT_OF_RANGE   = auto()
    MF_SOLVER_FAILED = auto()
    NUMERICAL        = auto()
    ACAUSAL          = auto()
    UNSTABLE         = auto()
    LOW_VARIATION    = auto()


def _rs_nat(kF, m): return _fs(kF, m)/np.pi**2
def _rs_fm(kF, m): return _fs(kF, m)/(np.pi**2*hc3)


def _lep(mu_e):
    ne = nmu = el = pl = 0.0
    if mu_e > m_e:
        k = np.sqrt(mu_e**2-m_e**2)
        ne = _n_from_kF(k); el += _fe(k,m_e)/(np.pi**2*hc3); pl += _fp(k,m_e)/(np.pi**2*hc3)
    if mu_e > m_mu:
        k = np.sqrt(mu_e**2-m_mu**2)
        nmu = _n_from_kF(k); el += _fe(k,m_mu)/(np.pi**2*hc3); pl += _fp(k,m_mu)/(np.pi**2*hc3)
    return ne, nmu, el, pl

def _newton_2d(residual_func, ms0, xp0, n_residuals=2,
               max_iter=15, tol=1e-8, h_ms=0.5, h_xp=1e-5):
    ms, xp = ms0, xp0
    for _ in range(max_iter):
        r = residual_func(ms, xp)
        if r is None:
            return None
        f1, f2 = r[0], r[1]
        if abs(f1) < tol and abs(f2) < tol:
            return ms, xp
        rp = residual_func(ms + h_ms, xp)
        rm = residual_func(ms - h_ms, xp)
        if rp is None or rm is None:
            return None
        J00 = (rp[0] - rm[0]) / (2 * h_ms)
        J10 = (rp[1] - rm[1]) / (2 * h_ms)
        rp = residual_func(ms, xp + h_xp)
        rm = residual_func(ms, xp - h_xp)
        if rp is None or rm is None:
            return None
        J01 = (rp[0] - rm[0]) / (2 * h_xp)
        J11 = (rp[1] - rm[1]) / (2 * h_xp)
        det = J00 * J11 - J01 * J10
        if abs(det) < 1e-30:
            return None
        dms = -(J11 * f1 - J01 * f2) / det
        dxp = -(-J10 * f1 + J00 * f2) / det
        alpha = 1.0
        for _ in range(5):
            ms_n = ms + alpha * dms
            xp_n = xp + alpha * dxp
            if 50 < ms_n < mN - 1 and 1e-6 < xp_n < 0.5:
                break
            alpha *= 0.5
        else:
            return None
        ms, xp = ms_n, xp_n
    r = residual_func(ms, xp)
    if r is None:
        return None
    if abs(r[0]) < 1e-5 and abs(r[1]) < 1e-4:
        return ms, xp
    return None


def _solve_mf_with_fallback(residual_func, ms_prev, xp_prev,
                            xp_trials=(0.04, 0.02, 0.08, 0.005)):
    sol = _newton_2d(residual_func, ms_prev, xp_prev)
    if sol is not None:
        return sol
    for xpt in xp_trials:
        for mst in [ms_prev, ms_prev * 0.9, ms_prev * 1.1]:
            if mst <= 10 or mst >= mN:
                continue
            sol = _newton_2d(residual_func, mst, xpt)
            if sol is not None:
                return sol
    try:
        def res_vec(y):
            r = residual_func(y[0], y[1])
            return [r[0], r[1]] if r is not None else [1e10, 1e10]
        s = root(res_vec, [ms_prev, xp_prev], method='hybr',
                 tol=1e-10, options={'maxfev': 300})
        if s.success and 10 < s.x[0] < mN and 0 < s.x[1] < 0.5:
            return (s.x[0], s.x[1])
    except Exception:
        pass
    return None


def _cs2_from_finite_diff(ea, pa):
    N = len(ea)
    cs2 = np.empty(N)
    for k in range(N):
        if k == 0:
            dp = pa[1] - pa[0]; de = ea[1] - ea[0]
        elif k == N - 1:
            dp = pa[-1] - pa[-2]; de = ea[-1] - ea[-2]
        else:
            dp = pa[k + 1] - pa[k - 1]; de = ea[k + 1] - ea[k - 1]
        if de <= 0:
            return None, Rejection.UNSTABLE
        cs2[k] = dp / de
    return cs2, None

def _solve_cubic_positive(a, b, rhs):
    p = b / a
    q = -rhs / a
    q_half = q * 0.5
    D = q_half * q_half + (p / 3.0) ** 3
    sqrt_D = np.sqrt(D)
    u = np.cbrt(-q_half + sqrt_D)
    return u - p / (3.0 * u) if abs(u) > 1e-30 else 0.0

def _omega0(gomg, nB, zeta):
    rhs = gomg * nB * hc3
    if zeta < 1e-15: return rhs / M_OMEGA**2
    a = zeta/6 * gomg**4
    return _solve_cubic_positive(a, M_OMEGA**2, rhs)

def _solve_vector_fields_CC(gw, gr, nB, ni, zeta, lam_w, max_iter=20, tol=1e-12):
    rhs_w = gw * nB * hc3
    rhs_r = gr * ni * hc3 / 2
    w0 = _omega0(gw, nB, zeta)

    a_cubic = zeta/6 * gw**4 if zeta >= 1e-15 else 0.0
    lw2 = 2*lam_w*gw**2*gr**2

    for _ in range(max_iter):
        mre2 = M_RHO**2 + lw2*w0**2
        r0 = rhs_r / mre2
        mw2_eff = M_OMEGA**2 + lw2*r0**2
        if a_cubic < 1e-30:
            w0_new = rhs_w / mw2_eff
        else:
            w0_new = _solve_cubic_positive(a_cubic, mw2_eff, rhs_w)

        if abs(w0_new - w0) < tol * (abs(w0) + 1e-20):
            mre2_final = M_RHO**2 + lw2*w0_new**2
            r0_final = rhs_r / mre2_final
            return w0_new, r0_final, mre2_final
        w0 = w0_new
    mre2 = M_RHO**2 + lw2*w0**2
    r0 = rhs_r / mre2
    return w0, r0, mre2

def _solve_ms(kFn, kFp, gs, ms_mass, b, c, guess=None):
    def res(ms):
        if ms<=1 or ms>=mN-1: return 1e10
        S=mN-ms; rs=_rs_nat(kFn,ms)+_rs_nat(kFp,ms)
        return ms_mass**2*S/gs**2 + b*mN*S**2 + c*S**3 - rs
    lo,hi = 50, mN-1
    try: return brentq(res, lo, hi, xtol=1e-8)
    except:
        try: return brentq(res, 100, mN-50, xtol=1e-6)
        except: return None

def _snm(nB, gs, gw, ms_mass, b, c, zeta):
    kF = _kF(nB/2)
    ms = _solve_ms(kF, kF, gs, ms_mass, b, c)
    if ms is None: return None
    S=mN-ms; s0=S/gs; U=(b*mN/3)*S**3+(c/4)*S**4
    ek = 2*_fe(kF,ms)/(np.pi**2*hc3)
    pk = 2*_fp(kF,ms)/(np.pi**2*hc3)
    es = (0.5*ms_mass**2*s0**2+U)/hc3; ps = -es
    w0 = _omega0(gw, nB, zeta)
    if w0 is None: return None
    W=gw*w0
    ev = (0.5*M_OMEGA**2*w0**2+3*zeta/24*W**4)/hc3
    pv = (0.5*M_OMEGA**2*w0**2+zeta/24*W**4)/hc3
    eps=ek+es+ev; P=pk+ps+pv; EA=eps/nB-mN
    return EA, P, ms


def _solve_sat_CC(K0, msr, J, msig, zeta, lam_w):
    mstar_t = msr*mN

    def residuals(x):
        gs,gw,bb,cc = x
        if gs<1 or gw<1: return [1e10]*4
        r = _snm(n0, gs, gw, msig, bb, cc, zeta)
        if r is None: return [1e10]*4
        EA,P,ms = r
        dn=n0*0.003
        rp = _snm(n0+dn, gs, gw, msig, bb, cc, zeta)
        rm = _snm(n0-dn, gs, gw, msig, bb, cc, zeta)
        if rp is None or rm is None: return [1e10]*4
        K0c = 9*n0**2*(rp[0]-2*EA+rm[0])/dn**2
        return [EA-E_SAT, P*10, (ms-mstar_t)/100, (K0c-K0)/100]

    best = None
    for g0 in [(10,12,0.003,-0.003),(8,10,0.005,-0.005),(12,14,0.001,-0.001),
               (9,11,0.01,-0.01),(11,13,0.002,0),(10,13,0,-0.005),
               (10,12,0.008,-0.002),(9,13,0.004,-0.004),(11,11,0.006,-0.006)]:
        try:
            sol = root(residuals, g0, method='hybr', tol=1e-8, options={'maxfev':3000})
            if sol.success and np.max(np.abs(sol.fun)) < 0.05:
                best = sol; break
        except: continue
    if best is None: return None

    gs,gw,bb,cc = best.x
    if gs<1 or gw<1: return None
    r = _snm(n0, gs, gw, msig, bb, cc, zeta)
    if r is None: return None
    EA,P,ms = r
    if abs(EA-E_SAT)>0.5 or abs(P)>0.5 or abs(ms-mstar_t)>5: return None

    # gρ from J
    kF = _kF(n0/2); EF = np.sqrt(kF**2+ms**2)
    Jk = kF**2/(6*EF); Jp = J-Jk
    if Jp<=0: return None
    w0 = _omega0(gw, n0, zeta)
    if w0 is None: return None
    if lam_w < 1e-15:
        gr2 = 8*Jp*M_RHO**2/(n0*hc3)
    else:
        den = n0*hc3 - 16*Jp*lam_w*gw**2*w0**2
        if den<=0: return None
        gr2 = 8*Jp*M_RHO**2/den
    if gr2<=0: return None
    gr = np.sqrt(gr2)

    # L
    dn=n0*0.005
    def Es(nB):
        kf=_kF(nB/2); ef=np.sqrt(kf**2+ms**2)
        jk=kf**2/(6*ef)
        wo=_omega0(gw,nB,zeta)
        if wo is None: return None
        mre=M_RHO**2+2*lam_w*gw**2*gr**2*wo**2
        return jk+gr**2*nB*hc3/(8*mre)
    ep=Es(n0+dn); em=Es(n0-dn)
    if ep is None or em is None: return None
    L=3*n0*(ep-em)/(2*dn)
    if L<L_MIN or L>L_MAX: return None
    return gs,gw,gr,bb,cc,L

def _eval_CC_density(nB, gs, gw, gr, msig, b, c, zeta, lam_w, ms_prev, xp_prev):
    def _residual(ms, xp):
        """Return (f1, f2, vf) or None if inputs out of bounds."""
        if ms<=10 or ms>=mN-1 or xp<1e-6 or xp>0.5: return None
        nn=(1-xp)*nB; np_=xp*nB
        kn=_kF(nn); kp=_kF(np_); S=mN-ms
        rn=_rs_nat(kn,ms)+_rs_nat(kp,ms)
        eq1 = msig**2*S/gs**2 + b*mN*S**2 + c*S**3 - rn
        ni=np_-nn
        vf=_solve_vector_fields_CC(gw,gr,nB,ni,zeta,lam_w)
        if vf is None: return None
        w0,r0,mre=vf
        en=np.sqrt(kn**2+ms**2); ep=np.sqrt(kp**2+ms**2)
        mun=en+gw*w0-0.5*gr*r0; mup=ep+gw*w0+0.5*gr*r0
        mu_e=mun-mup
        ne,nmu,_,_=_lep(mu_e)
        return eq1/1e6, (np_-ne-nmu)*1e3, vf

    sol = _solve_mf_with_fallback(_residual, ms_prev, xp_prev)
    if sol is None: return None

    ms,xp = sol
    if ms<=10 or ms>=mN or xp<0 or xp>0.5: return None
    nn=(1-xp)*nB; np_=xp*nB; kn=_kF(nn); kp=_kF(np_); S=mN-ms; s0=S/gs
    ek=(_fe(kn,ms)+_fe(kp,ms))/(np.pi**2*hc3)
    pk=(_fp(kn,ms)+_fp(kp,ms))/(np.pi**2*hc3)
    U=(b*mN/3)*S**3+(c/4)*S**4
    es=(0.5*msig**2*s0**2+U)/hc3; ps=-es
    ni=np_-nn
    vf=_solve_vector_fields_CC(gw,gr,nB,ni,zeta,lam_w)
    if vf is None: return None
    w0,r0,mre=vf
    W=gw*w0; R=gr*r0
    ev=(0.5*M_OMEGA**2*w0**2+3*zeta/24*W**4+0.5*M_RHO**2*r0**2+3*lam_w*W**2*R**2)/hc3
    pv=(0.5*M_OMEGA**2*w0**2+zeta/24*W**4+0.5*M_RHO**2*r0**2+lam_w*W**2*R**2)/hc3
    en=np.sqrt(kn**2+ms**2); ep_=np.sqrt(kp**2+ms**2)
    mu_e=(en+gw*w0-0.5*gr*r0)-(ep_+gw*w0+0.5*gr*r0)
    _,_,el,pl=_lep(mu_e)
    return ek+es+ev+el, pk+ps+pv+pl, ms, xp

def _evaluate_CC(gs,gw,gr,msig,b,c,zeta,lw,ms0):
    ea=np.empty(N_GRID); pa=np.empty(N_GRID)
    msp=ms0; xpp=0.04
    for i in range(N_GRID):
        r=_eval_CC_density(NB_GRID[i],gs,gw,gr,msig,b,c,zeta,lw,msp,xpp)
        if r is None: return None, Rejection.MF_SOLVER_FAILED
        ea[i],pa[i],msp,xpp=r
    return _cs2_from_finite_diff(ea, pa)

_DDME2 = {'a_sig':1.3881,'b_sig':1.0943,'c_sig':1.7057,'d_sig':0.4421,
           'a_omg':1.3892,'b_omg':0.9240,'c_omg':1.4620,'d_omg':0.4775,
           'a_rho':0.5647}

def _h_iso(x,a,b,c,d): xd=x+d; return a*(1+b*xd**2)/(1+c*xd**2)
def _h_rho(x,ar): return np.exp(-ar*(x-1))

def _eval_DD_density(nB, gs0, gw0, gr0, msig, ffs, ffw, ar, msp, xpp):
    x=nB/n0
    hs=_h_iso(x,*ffs); hw=_h_iso(x,*ffw); hr=_h_rho(x,ar)
    gs=gs0*hs; gw=gw0*hw; gr=gr0*hr
    dx=1e-4
    dhs=((_h_iso(x+dx,*ffs)-_h_iso(x-dx,*ffs))/(2*dx))/n0
    dhw=((_h_iso(x+dx,*ffw)-_h_iso(x-dx,*ffw))/(2*dx))/n0
    dgs=gs0*dhs; dgw=gw0*dhw; dgr=gr0*(-ar/n0)*hr

    def _residual_dd(ms, xp):
        if ms<=10 or ms>=mN-1 or xp<1e-6 or xp>0.5: return None
        nn=(1-xp)*nB; np_=xp*nB; kn=_kF(nn); kp=_kF(np_); S=mN-ms
        rn=_rs_nat(kn,ms)+_rs_nat(kp,ms)
        eq1=msig**2*S/gs**2-rn
        w0=gw*nB*hc3/M_OMEGA**2; ni=np_-nn; r0=gr*ni*hc3/(2*M_RHO**2)
        s0=S/gs; rfm=_rs_fm(kn,ms)+_rs_fm(kp,ms)
        Sr=dgw*gw*nB**2*hc3/M_OMEGA**2 - dgs*s0*rfm + dgr*gr*ni**2*hc3/(4*M_RHO**2)
        en=np.sqrt(kn**2+ms**2); ep=np.sqrt(kp**2+ms**2)
        mun=en+gw*w0-0.5*gr*r0+Sr; mup=ep+gw*w0+0.5*gr*r0+Sr
        mu_e=mun-mup; ne,nmu,_,_=_lep(mu_e)
        return eq1/1e6, (np_-ne-nmu)*1e3

    sol = _solve_mf_with_fallback(_residual_dd, msp, xpp,
                                  xp_trials=(0.04, 0.02, 0.08))
    if sol is None: return None

    ms,xp=sol
    if ms<=10 or ms>=mN or xp<0 or xp>0.5: return None
    nn=(1-xp)*nB; np_=xp*nB; kn=_kF(nn); kp=_kF(np_); S=mN-ms; s0=S/gs
    ek=(_fe(kn,ms)+_fe(kp,ms))/(np.pi**2*hc3)
    pk=(_fp(kn,ms)+_fp(kp,ms))/(np.pi**2*hc3)
    es=0.5*msig**2*s0**2/hc3; ps=-es
    w0=gw*nB*hc3/M_OMEGA**2; ni=np_-nn; r0=gr*ni*hc3/(2*M_RHO**2)
    ev=(0.5*M_OMEGA**2*w0**2+0.5*M_RHO**2*r0**2)/hc3; pv=ev
    rfm=_rs_fm(kn,ms)+_rs_fm(kp,ms)
    Sr=dgw*gw*nB**2*hc3/M_OMEGA**2-dgs*s0*rfm+dgr*gr*ni**2*hc3/(4*M_RHO**2)
    en=np.sqrt(kn**2+ms**2); ep_=np.sqrt(kp**2+ms**2)
    mu_e=(en+gw*w0-0.5*gr*r0+Sr)-(ep_+gw*w0+0.5*gr*r0+Sr)
    _,_,el,pl=_lep(mu_e)
    return ek+es+ev+el, pk+ps+pv+pl+nB*Sr, ms, xp

def _solve_sat_DD(K0, msr, J, msig, ffs, ffw, ar, P_tol=1.5):
    ms=msr*mN; S=mN-ms; kF=_kF(n0/2); EF=np.sqrt(kF**2+ms**2)
    rn=2*_rs_nat(kF,ms)
    if rn<=0: return None
    gs02=msig**2*S/rn
    if gs02<=0: return None
    gs0=np.sqrt(gs02); s0=S/gs0
    ek=2*_fe(kF,ms)/(np.pi**2*hc3); pk=2*_fp(kF,ms)/(np.pi**2*hc3)
    es=0.5*msig**2*s0**2/hc3
    ns_fm3=2*_rs_fm(kF,ms)
    ev_target = n0*(E_SAT + mN) - ek - es
    if ev_target <= 0: return None
    gw02 = 2*ev_target*M_OMEGA**2/(n0**2*hc3)
    if gw02<=0: return None
    gw0=np.sqrt(gw02)
    ev=0.5*gw02*n0**2*hc3/M_OMEGA**2
    dx_ff=1e-4
    dhs_dx=(_h_iso(1+dx_ff,*ffs)-_h_iso(1-dx_ff,*ffs))/(2*dx_ff)
    dhw_dx=(_h_iso(1+dx_ff,*ffw)-_h_iso(1-dx_ff,*ffw))/(2*dx_ff)
    Sr_omg = gw02*dhw_dx*n0*hc3/M_OMEGA**2
    Sr_sig = -(dhs_dx/n0)*S*ns_fm3
    P_sat = pk - es + ev + n0*(Sr_omg + Sr_sig)
    if abs(P_sat) > P_tol: return None
    Jk=kF**2/(6*EF); Jp=J-Jk
    if Jp<=0: return None
    gr02=8*Jp*M_RHO**2/(n0*hc3)
    if gr02<=0: return None
    gr0=np.sqrt(gr02)
    # L
    dn=n0*0.005
    def Es(nB):
        kf=_kF(nB/2); ef=np.sqrt(kf**2+ms**2)
        x=nB/n0; grl=gr0*_h_rho(x,ar)
        return kf**2/(6*ef)+grl**2*nB*hc3/(8*M_RHO**2)
    L=3*n0*(Es(n0+dn)-Es(n0-dn))/(2*dn)
    if L<L_MIN or L>L_MAX: return None
    return gs0,gw0,gr0,L

def _evaluate_DD(gs0,gw0,gr0,msig,ffs,ffw,ar,ms0):
    ea=np.empty(N_GRID); pa=np.empty(N_GRID)
    msp=ms0; xpp=0.04
    for i in range(N_GRID):
        r=_eval_DD_density(NB_GRID[i],gs0,gw0,gr0,msig,ffs,ffw,ar,msp,xpp)
        if r is None: return None, Rejection.MF_SOLVER_FAILED
        ea[i],pa[i],msp,xpp=r
    return _cs2_from_finite_diff(ea, pa)

def _dd_params(rng):
    def v(val):
        lo,hi=val*(1-DD_VARIATION),val*(1+DD_VARIATION)
        if lo>hi: lo,hi=hi,lo
        return uniform(lo,hi,rng=rng)
    fs=(v(_DDME2['a_sig']),v(_DDME2['b_sig']),v(_DDME2['c_sig']),v(_DDME2['d_sig']))
    fw=(v(_DDME2['a_omg']),v(_DDME2['b_omg']),v(_DDME2['c_omg']),v(_DDME2['d_omg']))
    ar=v(_DDME2['a_rho'])
    h1s=_h_iso(1,*fs); h1w=_h_iso(1,*fw)
    if abs(h1s)<1e-10 or abs(h1w)<1e-10: return None
    fs=(fs[0]/h1s,fs[1],fs[2],fs[3]); fw=(fw[0]/h1w,fw[1],fw[2],fw[3])
    x_test = np.linspace(1.0, 8.0, 30)
    hs_test = np.array([_h_iso(x, *fs) for x in x_test])
    hw_test = np.array([_h_iso(x, *fw) for x in x_test])
    if np.any(np.diff(hs_test) > 0) or np.any(np.diff(hw_test) > 0):
        return None
    return fs,fw,ar

def _s_CC_broad(rng=None):
    return('CC',uniform(K0_MIN,K0_MAX,rng=rng),uniform(MSTAR_RATIO_MIN,MSTAR_RATIO_MAX,rng=rng),
     uniform(J_MIN,J_MAX,rng=rng),uniform(MSIGMA_MIN,MSIGMA_MAX,rng=rng),
     uniform(ZETA_MIN,ZETA_MAX,rng=rng),uniform(LAMBDA_W_MIN,LAMBDA_W_MAX,rng=rng),None,None,None)
def _s_CC_stiff(rng=None):
    return('CC',uniform(250,K0_MAX,rng=rng),uniform(MSTAR_RATIO_MIN,0.65,rng=rng),
     uniform(J_MIN,J_MAX,rng=rng),uniform(MSIGMA_MIN,MSIGMA_MAX,rng=rng),
     uniform(ZETA_MIN,0.02,rng=rng),uniform(LAMBDA_W_MIN,LAMBDA_W_MAX,rng=rng),None,None,None)
def _s_CC_soft(rng=None):
    return('CC',uniform(K0_MIN,240,rng=rng),uniform(0.70,MSTAR_RATIO_MAX,rng=rng),
     uniform(J_MIN,J_MAX,rng=rng),uniform(MSIGMA_MIN,MSIGMA_MAX,rng=rng),
     uniform(0.03,ZETA_MAX,rng=rng),uniform(LAMBDA_W_MIN,LAMBDA_W_MAX,rng=rng),None,None,None)
def _s_CC_high_zeta(rng=None):
    return('CC',uniform(K0_MIN,K0_MAX,rng=rng),uniform(MSTAR_RATIO_MIN,MSTAR_RATIO_MAX,rng=rng),
     uniform(J_MIN,J_MAX,rng=rng),uniform(MSIGMA_MIN,MSIGMA_MAX,rng=rng),
     uniform(0.03,ZETA_MAX,rng=rng),uniform(LAMBDA_W_MIN,LAMBDA_W_MAX,rng=rng),None,None,None)
def _s_CC_no_zeta(rng=None):
    return('CC',uniform(K0_MIN,K0_MAX,rng=rng),uniform(MSTAR_RATIO_MIN,MSTAR_RATIO_MAX,rng=rng),
     uniform(J_MIN,J_MAX,rng=rng),uniform(MSIGMA_MIN,MSIGMA_MAX,rng=rng),
     0.0,uniform(LAMBDA_W_MIN,LAMBDA_W_MAX,rng=rng),None,None,None)
def _s_CC_normal(rng=None):
    return('CC',normal_truncated(230,20,K0_MIN,K0_MAX,rng=rng),
     normal_truncated(0.65,0.06,MSTAR_RATIO_MIN,MSTAR_RATIO_MAX,rng=rng),
     normal_truncated(31.7,1.5,J_MIN,J_MAX,rng=rng),
     normal_truncated(500,50,MSIGMA_MIN,MSIGMA_MAX,rng=rng),
     uniform(ZETA_MIN,ZETA_MAX,rng=rng),uniform(LAMBDA_W_MIN,LAMBDA_W_MAX,rng=rng),None,None,None)
def _s_CC_beta(rng=None):
    return('CC',beta_scaled(2,2,K0_MIN,K0_MAX,rng=rng),
     beta_scaled(2,2,MSTAR_RATIO_MIN,MSTAR_RATIO_MAX,rng=rng),
     beta_scaled(2,2,J_MIN,J_MAX,rng=rng),beta_scaled(2,2,MSIGMA_MIN,MSIGMA_MAX,rng=rng),
     uniform(ZETA_MIN,ZETA_MAX,rng=rng),uniform(LAMBDA_W_MIN,LAMBDA_W_MAX,rng=rng),None,None,None)
def _s_CC_high_J(rng=None):
    return('CC',uniform(K0_MIN,K0_MAX,rng=rng),uniform(MSTAR_RATIO_MIN,MSTAR_RATIO_MAX,rng=rng),
     uniform(32,J_MAX,rng=rng),uniform(MSIGMA_MIN,MSIGMA_MAX,rng=rng),
     uniform(ZETA_MIN,ZETA_MAX,rng=rng),uniform(LAMBDA_W_MIN,0.02,rng=rng),None,None,None)
def _s_DD_broad(rng=None):
    dd=_dd_params(rng)
    if dd is None: return None
    return('DD',uniform(K0_MIN,K0_MAX,rng=rng),uniform(MSTAR_RATIO_MIN,MSTAR_RATIO_MAX,rng=rng),
     uniform(J_MIN,J_MAX,rng=rng),uniform(MSIGMA_MIN,MSIGMA_MAX,rng=rng),
     0.0,0.0,dd[0],dd[1],dd[2])
def _s_DD_normal(rng=None):
    dd=_dd_params(rng)
    if dd is None: return None
    return('DD',normal_truncated(240,15,K0_MIN,K0_MAX,rng=rng),
     normal_truncated(0.66,0.05,MSTAR_RATIO_MIN,MSTAR_RATIO_MAX,rng=rng),
     normal_truncated(32.3,1,J_MIN,J_MAX,rng=rng),
     normal_truncated(550,40,MSIGMA_MIN,MSIGMA_MAX,rng=rng),0.0,0.0,dd[0],dd[1],dd[2])

STRATEGIES=[_s_CC_broad,_s_CC_stiff,_s_CC_soft,_s_CC_high_zeta,_s_CC_no_zeta,
            _s_CC_normal,_s_CC_beta,_s_CC_high_J,_s_DD_broad,_s_DD_normal]


def generate_one_sample(rng=None):
    if rng is None:
        from sampling_utils import get_rng; rng=get_rng()
    strategy=STRATEGIES[rng.integers(len(STRATEGIES))]
    params=strategy(rng=rng)
    if params is None: return None,strategy.__name__,Rejection.COUPLING_FAILED
    variant=params[0]; K0,msr,J,msig,zeta,lw=params[1:7]
    ffs,ffw,ar=params[7:10]; ms0=msr*mN
    if variant=='CC':
        cp=_solve_sat_CC(K0,msr,J,msig,zeta,lw)
        if cp is None: return None,strategy.__name__,Rejection.COUPLING_FAILED
        gs,gw,gr,bb,cc,L=cp
        cs2,reason=_evaluate_CC(gs,gw,gr,msig,bb,cc,zeta,lw,ms0)
    elif variant=='DD':
        cp=_solve_sat_DD(K0,msr,J,msig,ffs,ffw,ar)
        if cp is None: return None,strategy.__name__,Rejection.COUPLING_FAILED
        gs0,gw0,gr0,L=cp
        cs2,reason=_evaluate_DD(gs0,gw0,gr0,msig,ffs,ffw,ar,ms0)
    else: return None,strategy.__name__,Rejection.COUPLING_FAILED
    if cs2 is None: return None,strategy.__name__,reason
    tag=cs2_acceptance_check(cs2)
    if tag is not None: return None,strategy.__name__,Rejection[tag]
    return cs2,strategy.__name__,None
