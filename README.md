# BPS-AKT

## Setups
The requiring environment is as bellow:  

- Linux 
- Python 3+
- PyTorch 1.2.0 
- Scikit-learn 0.21.3
- Scipy 1.3.1
- Numpy 1.17.2



## Running BPS-AKT.
Here are some examples for using BPS_AKT model (with two side factors x and y):

(on ASSISTments2009 and ASSISTments2017 datasets)
```
python main_bps.py --dataset assist2009_eid --model akt_eid_xid_yid
python main_bps.py --dataset assist2017_eid --model akt_eid_xid_yid
```

## Running RFE-AKT.
Here are some examples for using REF_AKT model:

(on ASSISTments2009 and ASSISTments2017 datasets)
```
python main_pfe.py --dataset assist2009_eid --model akt_eid_xid_yid
python main_pfe.py --dataset assist2017_eid --model akt_eid_xid_yid
```
## Running RME-AKT.
Here are some examples for using BPS_AKT model:

(on ASSISTments2009 and ASSISTments2017 datasets)
```
python main_pfe.py --dataset assist2009_eid --model akt_eid_xid_yid
python main_pfe.py --dataset assist2017_eid --model akt_eid_xid_yid
```
