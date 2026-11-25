# Cover letter

**Manuscript title:** Matched-Capacity Hybrid Quantum-Classical Networks for Confidence Calibration under Cross-Population Chest X-Ray Shift

**Author:** Tran Thi Thuy Quyen, Information Assurance, FPT University Can Tho, Vietnam

**Intended venues (in order):** Scientific Reports; Computers in Biology and Medicine; Quantum Information Processing. Not IEEE TNNLS.

Dear Editor,

Please consider this manuscript. It reports a negative but controlled result: on a shared six-dimensional bottleneck from a frozen DenseNet-121, a parameterized quantum circuit does not improve confidence calibration relative to a capacity-matched MLP when CheXpert-trained heads are evaluated on VinDr-CXR.

The comparison is matched on purpose. The PQC never sees pixels. Product and entangling circuits differ only by a CNOT ring. Pneumonia is not claimed across datasets because VinDr-CXR has no Pneumonia class. Temperature scaling is fitted on CheXpert validation logits and tested for transfer; it can lower source ECE and raise target ECE on the same run.

We believe this is useful to readers of calibration, medical imaging shift, and variational quantum models precisely because it refuses unmatched DenseNet baselines and ontology collapse. The limitations (six qubits, simulator, N small, three seeds) are stated in the main text.

The work is original, not under consideration elsewhere, and uses only public de-identified datasets (CheXpert, VinDr-CXR).

Sincerely,  
Tran Thi Thuy Quyen
