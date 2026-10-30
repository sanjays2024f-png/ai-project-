"""Synthetic india_dataset.csv generator (Step 1 stand-in for NASA).
Writes data/raw/synthetic/india_dataset.csv by default (synthetic provenance)."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
BPS=[30,60,90,120,250]
CLS=["Good","Satisfactory","Moderate","Poor","Very Poor","Severe"]
# district,state,zone,lat,lon,elev,pop
DISTS=[
("Central Delhi","Delhi","IGP",28.61,77.23,216,29000),
("Lucknow","Uttar Pradesh","IGP",26.85,80.95,123,2500),
("Patna","Bihar","IGP",25.60,85.13,53,3200),
("Ludhiana","Punjab","IGP",30.90,75.85,262,2100),
("Kolkata","West Bengal","East",22.57,88.36,9,24000),
("Bhubaneswar","Odisha","East",20.27,85.84,45,2100),
("Itanagar","Arunachal Pradesh","Northeast",27.10,93.62,440,300),
("Guwahati","Assam","Northeast",26.14,91.73,55,1800),
("Shillong","Meghalaya","Northeast",25.57,91.88,1525,800),
("Gangtok","Sikkim","Northeast",27.33,88.61,1650,400),
("Bengaluru Urban","Karnataka","South",12.97,77.59,920,4400),
("Maheswaram","Telangana","South",17.13,78.03,545,800),
("Kanyakumari","Tamil Nadu","South",8.08,77.53,19,1100),
("Chennai","Tamil Nadu","South",13.08,80.27,7,14000),
("Bhuj","Gujarat","West",23.24,69.66,110,300),
("Jaisalmer","Rajasthan","West",26.91,70.91,225,50),
("Mumbai","Maharashtra","West",19.07,72.87,14,21000),
("Bhopal","Madhya Pradesh","Central",23.25,77.40,527,1500),
("Nagpur","Maharashtra","Central",21.14,79.08,310,1900),
("Srinagar","Jammu and Kashmir","Himalaya",34.08,74.79,1585,1400),
("Leh","Ladakh","Himalaya",34.15,77.58,3500,10),
("Shimla","Himachal Pradesh","Himalaya",31.10,77.17,2276,500),
("Port Blair","Andaman and Nicobar","Islands",11.62,92.93,16,300),
("Kavaratti","Lakshadweep","Islands",10.56,72.64,2,2000),
]
ZBASE={"IGP":85,"East":55,"Northeast":35,"South":38,"West":50,"Central":52,"Himalaya":28,"Islands":18}
def aqi(pm: float) -> str:
    for bp,c in zip(BPS,CLS):
        if pm<=bp: return c
    return "Severe"
def generate(n_days: int = 180, seed: int = 42) -> "pd.DataFrame":
    """SYNTHETIC stand-in ONLY (smoke tests). Provenance: synthetic_generator_v1."""
    rng=np.random.default_rng(seed)
    dates=pd.date_range("2024-08-01",periods=n_days,freq="D")
    rows=[];cid=0
    for di,st,zo,la0,lo0,el,pp in DISTS:
        for _ in range(2):
            cid+=1
            la=la0+float(rng.normal(0,0.06));lo=lo0+float(rng.normal(0,0.06))
            base=ZBASE[zo]
            for dt in dates:
                m=int(dt.month);w=1.0 if m in (11,12,1,2) else 0.0
                mo=1.0 if m in (6,7,8,9) else 0.0
                dw=str(dt.date()) in ("2024-10-31","2024-11-01","2024-11-02")
                diw=1.0 if (dw and zo in ("IGP","Central","West")) else 0.0
                aod=max(0.05,float(rng.normal(0.45+0.35*w-0.15*mo,0.18)))
                no2=max(1.0,float(rng.normal(18+10*w+(8 if zo=="IGP" else 0),6)))
                so2=max(0.5,float(rng.normal(8+3*w,3)))
                co=max(50.0,float(rng.normal(350+150*w+120*diw,80)))
                bc=max(0.2,float(rng.normal(2.5+1.5*w+1.0*diw,0.9)))
                du=max(0.1,float(rng.normal(3.0+(6.0 if di=="Jaisalmer" else 0),1.5)))
                tp=float(rng.normal(28-10*w-(12 if zo=="Himalaya" else 0),2.5))
                rh=float(np.clip(rng.normal(65+15*mo-15*w,10),10,98))
                wu=float(rng.normal(1.5,2.0));wv=float(rng.normal(0.5,2.0))
                blh=max(100.0,float(rng.normal(600-200*w,150)))
                rn=float(np.clip(rng.exponential(2.0+8.0*mo),0,120))
                fm=0.3
                if di in ("Ludhiana","Patna","Central Delhi") and m in (10,11): fm=4.0
                if zo=="Northeast" and m in (3,4): fm=2.5
                fc=int(rng.poisson(fm))
                nd=float(np.clip(rng.normal(0.45+0.15*mo-0.1*w,0.12),0.05,0.9))
                lst=tp+float(rng.normal(3,2))
                pm=(base+45*aod+0.6*no2+0.02*co+3.0*bc+1.5*du+0.004*pp
                    +35*w-18*mo+55*diw+2.0*fc-0.4*rn-0.01*blh+float(rng.normal(0,10)))
                pm=float(np.clip(pm,5,450))
                rows.append({"date":dt.date().isoformat(),"cell_id":cid,"lat":round(la,4),
                 "lon":round(lo,4),"district":di,"state":st,"zone":zo,"aod":round(aod,3),
                 "no2":round(no2,2),"so2":round(so2,2),"co":round(co,1),"bc":round(bc,2),
                 "dust":round(du,2),"temp":round(tp,2),"rh":round(rh,1),
                 "wind_u":round(wu,2),"wind_v":round(wv,2),"blh":round(blh,1),
                 "rain":round(rn,2),"fire_count":fc,"ndvi":round(nd,3),"lst":round(lst,2),
                 "elevation":el,"population":pp,"pm25":round(pm,1),"aqi_class":aqi(pm),
                 "disease_risk":"High" if pm>90 else "Medium" if pm>45 else "Low",
                 "provenance":"synthetic_generator_v1"})
    df=pd.DataFrame(rows)
    r=np.random.default_rng(seed+1)
    mh=(df.zone=="Himalaya")&(r.random(len(df))<0.12)
    mt=(df.district=="Jaisalmer")&(r.random(len(df))<0.10)
    mm=(df.zone.isin(["Northeast","South"]))&(r.random(len(df))<0.08)
    df.loc[mh|mt|mm,"aod"]=np.nan
    df.loc[mh,"ndvi"]=np.nan
    return df
def main() -> None:
    import argparse
    from pathlib import Path
    ap=argparse.ArgumentParser(description="Generate synthetic india_dataset.csv")
    ap.add_argument("--n_days",type=int,default=180)
    ap.add_argument("--out",type=str,default="data/raw/synthetic/india_dataset.csv")
    ap.add_argument("--seed",type=int,default=42)
    a=ap.parse_args()
    df=generate(n_days=a.n_days,seed=a.seed)
    o=Path(a.out);o.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(o,index=False, encoding="utf-8")
    print(f"Wrote {len(df)} rows x {len(df.columns)} cols -> {o}")
    print("Zones:",df.zone.value_counts().to_dict())
    print("AQI:",df.aqi_class.value_counts().to_dict())
if __name__=="__main__":
    main()


