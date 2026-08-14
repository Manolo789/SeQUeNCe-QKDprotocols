
## + QKD protocols for the SeQUENCe simulator. 

This extension to the SeQUENCe simulator adds the **B92**, **COW**, **BBM92** and **E91** quantum key distribution protocols. Additionally, it creates mechanisms to compare the **BB84**, **B92**, **COW**, **BBM92** and **E91** protocols based on performance parameters under different simulation scenarios.
This extension emerged from the scientific initiation project 'Simulation and Testing for Performance Analysis of Communication Protocols in Quantum Networks' at the Laboratory of Computer Architecture and Networks ([LARC](https://www.larc.usp.br/) in Portuguese), a laboratory of the Department of Computer Engineering and Digital Systems of the Polytechnic School of the University of São Paulo (PCS-EPUSP), using resources from the Unified Scholarship Program ([PUB - Programa Unificado de Bolsas](https://prip.usp.br/apoio-estudantil/pub/)).



## Quantum channel in free space (QC in FS)

This repository has been extended with support for **free-space (FS) quantum communication** in the SeQUeNCe simulation framework.

The implementation includes a free-space optical channel model for QKD protocols, allowing the simulation of atmospheric propagation effects. Channel conditions can be parameterized using meteorological data and the atmospheric refractive index structure parameter \($$C_n^2$$\), enabling performance evaluation under different turbulence regimes.

## Implementation of Entanglement Protocols

The implementation of **BBM92** and **E91** extends the simulator with entanglement-based QKD protocols, enabling the evaluation of quantum communication systems that rely on entangled photon pairs. The **E91** implementation also supports the analysis of Bell inequality (CHSH) violations, allowing security assessment through quantum correlations.

### Link geometry and key-size-driven operation

For **every** protocol in the registry, `distance` is the **total Alice–Bob
separation**. For the entanglement-based protocols the untrusted source
(Charlie) sits between the two parties and the arms follow from
`charlie_position`:

```
distance = distance_ac + distance_cb
distance_ac = charlie_position * distance
distance_cb = (1 - charlie_position) * distance
```

Setting `charlie_position=None` takes the two arms from the optional
`distance_ac` / `distance_cb` parameters instead (asymmetric links given
directly rather than as a fraction).

The entanglement-based protocols are **key-size driven**, exactly like the
prepare-and-measure ones: `push(keysize, key_num, run_time)` starts the
generation, the sifted bits of successive emission trains accumulate in
`key_bits`, and a key is extracted whenever `len(key_bits) >= keysize`,
repeating until the runtime or `key_num` is exhausted. There is no "number of
rounds" knob — the emission rounds are a *derived* quantity (reported as
`num_rounds` for diagnostics). As a result both families are post-processed by
the **same** secure-key-rate estimator with the **same** denominator, so `R_s`
and `R_sk` are expressed in bits per qubit sent for all five protocols and are
directly comparable.

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

## Papers that Used and/or Extended `SeQUeNCe-QKDprotocols`
| Year | Authors | Title | Venue | Code |
|------|---------|-------|-------|------|
| 2026 |  CABRERA, Emanuel M. et al | [Protocolos de Comunicação Quântica: Estudo de Caso Prévio para Implementação de Enlace Experimental.](https://doi.org/10.5753/wqunets.2026.23453) | III WQuNets (SBRC 2026) | [GitHub](https://github.com/Manolo789/SeQUeNCe-QKDprotocols/releases/tag/v0.1) |


