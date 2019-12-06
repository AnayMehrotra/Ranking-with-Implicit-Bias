# Ranking-with-Implicit-Bias

This repository provides a copy of the 2009 IIT-JEE Scores Dataset released in response to a Right to Information application filed in June 2009 [2]. We use this dataset to empirically evaluate the constraints to mitigate implicit bias in rankings proposed in our FAT*'20 paper [1].


### Parsing the dataset (using Python 3.*)
```
from meza import io
record = io.read('jee2009.mdb')
result=[]
for i in range(384977): 
    result.append(next(record))
```

### References
[1] *Interventions for Ranking in the Presence of Implicit Bias*
L. Elisa Celis, Anay Mehrotra, Nisheeth K. Vishnoi
ACM Conference on Fairness, Accountability, and Transparency, **ACM FAT* 2020**

[2] *RTI Complaint. Decision No. CIC/SG/C/2009/001088/5392, Complaint No. CIC/SG/C/2009/001088*
Rajeev Kumar, **2009**


