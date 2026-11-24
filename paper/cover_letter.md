# Cover letter

**Manuscript title:** What a Six-Qubit Head Can and Cannot Change: Calibration under Cross-Population Chest X-Ray Shift at Matched Capacity

**Author:** Tran Thi Thuy Quyen, Information Assurance, FPT University Can Tho, Vietnam

**Intended venues (in order):** Scientific Reports; Computers in Biology and Medicine; Quantum Information Processing. Not IEEE TNNLS.

Dear Editor,

Please consider this manuscript. It reports a negative but controlled result: on a shared six-dimensional bottleneck from a frozen DenseNet-121, a parameterized quantum circuit does not improve confidence calibration relative to a matched MLP when CheXpert-trained heads are evaluated on VinDr-CXR.

The comparison is matched on purpose. The PQC never sees pixels, and all heads are trained identically. A key methodological point is stated openly: with a single-qubit readout and no entangling gates, the product PQC is functionally a one-feature model, so nominal parameter parity does not imply functional parity; the clean classical-versus-quantum comparison is MLP versus entangling PQC. Pneumonia is not claimed across datasets because VinDr-CXR has no Pneumonia class. Temperature scaling is fitted on CheXpert validation logits and tested for transfer; it can lower source ECE and raise target ECE on the same run. Calibration is reported on both balanced and natural-prevalence target holdouts, and the full metric set (AUROC, AUPRC, F1, Brier, ECE, MCE) is reported. The limitations (six qubits, simulator, small N, three seeds, fixed bottleneck) are stated in the main text.

The work is original, not under consideration elsewhere, and uses only public de-identified datasets (CheXpert, VinDr-CXR).

Sincerely,  
Tran Thi Thuy Quyen