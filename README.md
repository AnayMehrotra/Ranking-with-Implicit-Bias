# Code for "Interventions for Ranking in the Presence of Implicit Bias"

This repository contains the code for reproducing the simulations in our paper

**Interventions for Ranking in the Presence of Implicit Bias**<br>
*L. Elisa Celis, Anay Mehrotra, Nisheeth K. Vishnoi*<br>
ACM Conference on Fairness, Accountability, and Transparency, **ACM FAT\* 2020**
arXiv: https://arxiv.org/abs/2011.04219


## Running Simulations
To run the simulations, inside the `simulations` folder, execute
```
python experiments.py
```
To generate other additional plots in the paper, inside the `simulations` folder, execute
```
python jee2009-additional-figs.py
python semantic-scholar-additional-figs.py
```

## 2009 IIT-JEE Scores Dataset 
This repository also provides a copy of the 2009 IIT-JEE Scores Dataset released in response to a Right to Information application filed in June 2009 [1]. We use this dataset in our empirical evaluations.


#### Parsing the dataset (using Python 3.*)
```
from meza import io
record = io.read('jee2009.mdb')
result=[]
for i in range(384977): 
    result.append(next(record))
```

## References
[1] *RTI Complaint. Decision No. CIC/SG/C/2009/001088/5392, Complaint No. CIC/SG/C/2009/001088*
Rajeev Kumar, **2009**
