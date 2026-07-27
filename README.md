
## + QKD protocols for the SeQUENCe simulator. 

This extension to the SeQUENCe simulator adds the **B92**, **COW**, **BBM92** and **E91** quantum key distribution protocols. Additionally, it creates mechanisms to compare the **BB84**, **B92**, **COW**, **BBM92** and **E91** protocols based on performance parameters under different simulation scenarios.
This extension emerged from the scientific initiation project 'Simulation and Testing for Performance Analysis of Communication Protocols in Quantum Networks' at the Laboratory of Computer Architecture and Networks ([LARC](https://www.larc.usp.br/) in Portuguese), a laboratory of the Department of Computer Engineering and Digital Systems of the Polytechnic School of the University of São Paulo (PCS-EPUSP), using resources from the Unified Scholarship Program ([PUB - Programa Unificado de Bolsas](https://prip.usp.br/apoio-estudantil/pub/)).



## Quantum channel in free space (QC in FS)

This repository has been extended with support for **free-space (FS) quantum communication** in the SeQUeNCe simulation framework.

The implementation includes a free-space optical channel model for QKD protocols, allowing the simulation of atmospheric propagation effects. Channel conditions can be parameterized using meteorological data and the atmospheric refractive index structure parameter \($$C_n^2$$\), enabling performance evaluation under different turbulence regimes.

## Implementation of Entanglement Protocols

The implementation of **BBM92** and **E91** extends the simulator with entanglement-based QKD protocols, enabling the evaluation of quantum communication systems that rely on entangled photon pairs. The **E91** implementation also supports the analysis of Bell inequality (CHSH) violations, allowing security assessment through quantum correlations.

## Features

- Implementation of the **B92**, **COW**, **BBM92**, and **E91** QKD protocols.
- Performance comparison among **BB84**, **B92**, **COW**, **BBM92**, and **E91**.
- Support for entanglement-based QKD simulations.
- CHSH Bell inequality analysis for the E91 protocol.
- Free-space optical channel for QKD simulations.
- Atmospheric turbulence modeling based on \(C_n^2\).
- Integration with existing SeQUeNCe QKD protocol implementations.
- Configurable atmospheric and link parameters.


## Installing
SeQUeNCe requires [Python](https://www.python.org/downloads/) 3.11 or later. This version of SeQUENCe has been modified in some aspects, therefore it is not yet available in the pip manager.


To install the modified simulator and run the extension, proceed as follows:
```
git clone https://github.com/Manolo789/SeQUeNCe-QKDprotocols.git
cd SeQUeNCe-QKDprotocols/simulator/
python3 -m venv .venv
pip install --break-system-packages --editable .
cd ..
python3 QKD_Extension.py
```

## Contributions
- The meteorological data used in `sensores/estação-solar-usp_Tabela01.dat` were provided by ([Prof. Dr. André Luiz Veiga Gimenes](https://www.linkedin.com/in/andr%C3%A9-gimenes-87044517)).


## Contact
If you have questions, please contact Emanuel at [em7411081@gmail.com](mailto:em7411081@gmail.com).


