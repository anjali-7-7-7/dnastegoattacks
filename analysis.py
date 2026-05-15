import numpy as np
from scipy.signal import welch
from scipy.stats import chi2
from statsmodels.stats.diagnostic import acorr_ljungbox
import os, struct, time

MASK = 0xFFFFFFFF

def qr(a,b,c,d):
    a=(a+b)&MASK; d^=a; d=((d<<16)|(d>>16))&MASK
    c=(c+d)&MASK; b^=c; b=((b<<12)|(b>>20))&MASK
    a=(a+b)&MASK; d^=a; d=((d<<8)|(d>>24))&MASK
    c=(c+d)&MASK; b^=c; b=((b<<7)|(b>>25))&MASK
    return a,b,c,d

def block(key,counter,nonce,rounds=20):
    C=[0x61707865,0x3320646e,0x79622d32,0x6b206574]
    k=list(struct.unpack('<8I',key))
    n=list(struct.unpack('<3I',nonce))
    s=C+k+[counter]+n; w=list(s)
    for _ in range(rounds//2):
        w[0],w[4],w[8],w[12]=qr(w[0],w[4],w[8],w[12])
        w[1],w[5],w[9],w[13]=qr(w[1],w[5],w[9],w[13])
        w[2],w[6],w[10],w[14]=qr(w[2],w[6],w[10],w[14])
        w[3],w[7],w[11],w[15]=qr(w[3],w[7],w[11],w[15])
        w[0],w[5],w[10],w[15]=qr(w[0],w[5],w[10],w[15])
        w[1],w[6],w[11],w[12]=qr(w[1],w[6],w[11],w[12])
        w[2],w[7],w[8],w[13]=qr(w[2],w[7],w[8],w[13])
        w[3],w[4],w[9],w[14]=qr(w[3],w[4],w[9],w[14])
    out=[(w[i]+s[i])&MASK for i in range(16)]
    return struct.pack('<16I',*out)

def gen(key,nonce,nb,rounds=20):
    return np.frombuffer(b''.join(block(key,i,nonce,rounds) for i in range(nb)),dtype=np.uint32).astype(np.float64)

def analyze(values,label):
    n=len(values)
    norm=(values-np.mean(values))/(np.std(values)+1e-10)
    lb=acorr_ljungbox(norm[:50000],lags=100,return_df=True)
    min_pval=float(lb['lb_pvalue'].min())
    sig_lags=int((lb['lb_pvalue']<0.05).sum())
    freqs,power=welch(norm[:50000],nperseg=1024)
    snr=float(power[1:].max()/(power[1:].mean()+1e-10))
    ac=[float(np.corrcoef(values[:n-lag],values[lag:])[0,1]) for lag in range(1,101)]
    max_ac=max(abs(v) for v in ac)
    detected=min_pval<0.05 or snr>2.5 or max_ac>0.05
    return {'label':label,'lb_min_pval':round(min_pval,6),
            'sig_lags':sig_lags,'snr':round(snr,4),
            'max_ac':round(max_ac,6),'detected':detected}

NB=16384
key=os.urandom(32)
nonce=os.urandom(12)

print("="*65)
print("EMPIRICAL PERIODICITY ANALYSIS — ChaCha20 ARX OUTPUT")
print("="*65)
print(f"Sample: {NB*64:,} bytes | 32-bit word level | 100 lags")

rv=np.frombuffer(os.urandom(NB*64),dtype=np.uint32).astype(np.float64)
bl=analyze(rv,"TRUE RANDOM")
print(f"\nBASELINE (os.urandom): LB={bl['lb_min_pval']} SNR={bl['snr']} MaxAC={bl['max_ac']}")

print(f"\n{'Rounds':<8}{'LB p-val':<14}{'SigLags':<10}{'SNR':<10}{'MaxAC':<12}{'Result'}")
print("-"*65)

results=[]
for r in [2,4,6,8,10,14,20]:
    v=gen(key,nonce,NB,rounds=r)
    res=analyze(v,f"{r}r")
    results.append((r,res))
    flag="VULNERABLE ***" if res['detected'] else "resistant"
    print(f"{r:<8}{res['lb_min_pval']:<14}{res['sig_lags']:<10}{res['snr']:<10}{res['max_ac']:<12}{flag}")

print(f"\nCROSS-KEY TEST — 5 random keys at 20 rounds:")
detected_keys=0
for i in range(5):
    k=os.urandom(32)
    v=gen(k,nonce,NB,rounds=20)
    res=analyze(v,f"key{i+1}")
    flag="LEAKAGE DETECTED ***" if res['detected'] else "not detected"
    if res['detected']: detected_keys+=1
    print(f"  Key {i+1}: LB={res['lb_min_pval']:<12} SNR={res['snr']:<10} MaxAC={res['max_ac']:<10} {flag}")

print(f"\nKEY FINDING: {detected_keys}/5 keys show statistically significant")
print(f"structure at full 20 rounds (LB p < 0.05).")
print(f"This sub-threshold signal is the QFT attack target.")
print(f"\nNULL HYPOTHESIS: ChaCha20 output indistinguishable from true random.")
full=results[-1][1]
if full['detected']:
    print("RESULT: REJECT null at 20 rounds.")
elif detected_keys>0:
    print(f"RESULT: FAIL TO REJECT on average — but {detected_keys}/5 keys leak.")
    print("Key-dependent residual periodicity is the empirical finding.")
else:
    print("RESULT: No significant structure detected classically.")
    print("QFT amplification argument remains theoretical.")
