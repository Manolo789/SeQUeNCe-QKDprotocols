
## + QKD protocols for the SeQUENCe simulator. 

This extension to the SeQUENCe simulator adds the B92 and COW protocols. Additionally, it creates mechanisms to compare the BB84, B92, and COW protocols based on performance parameters.
This extension emerged from the scientific initiation project 'Simulation and Testing for Performance Analysis of Communication Protocols in Quantum Networks' at the Laboratory of Computer Architecture and Networks ([LARC](https://www.larc.usp.br/) in Portuguese), a laboratory of the Department of Computer Engineering and Digital Systems of the Polytechnic School of the University of São Paulo (PCS-EPUSP), using resources from the Unified Scholarship Program ([PUB - Programa Unificado de Bolsas](https://prip.usp.br/apoio-estudantil/pub/)).


# Quantum channel in free space (QC in FS)
This branch is responsible for the development of a quantum channel model implementation that considers free space links. *The code is under development!*

## Installing
SeQUeNCe requires [Python](https://www.python.org/downloads/) 3.11 or later. This version of SeQUENCe has been modified in some aspects, therefore it is not yet available in the pip manager.


To install the modified simulator and run the extension, proceed as follows:
```
git clone https://github.com/Manolo789/SeQUeNCe-QKDprotocols.git
cd SeQUeNCe-QKDprotocols/simulator/
pip install miepython
python3 -m venv .venv
pip install --break-system-packages --editable .
cd ..
python3 QKD_Extension.py
```


## Contact
If you have questions, please contact Emanuel at [em7411081@gmail.com](mailto:em7411081@gmail.com).


