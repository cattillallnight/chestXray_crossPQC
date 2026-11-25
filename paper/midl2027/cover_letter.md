# MIDL 2027 Main Track — submission note

**Manuscript title:** Matched-Capacity Hybrid Quantum-Classical Networks for Confidence Calibration under Cross-Population Chest X-Ray Shift

**Track:** Main Track — full paper (PMLR), not short paper  
**Author:** Tran Thi Thuy Quyen  
Information Assurance, FPT University Can Tho, Vietnam  
quyentttiace200993@gmail.com

Dear Program Chairs,

Please consider this full paper for the MIDL 2027 Main Track. It reports a negative but controlled result: on a shared six-dimensional bottleneck from a frozen ImageNet DenseNet-121, a standard parameterized quantum circuit does not improve confidence calibration relative to a capacity-matched MLP when CheXpert-trained heads are evaluated on VinDr-CXR.

The manuscript does not propose a new architecture and does not claim state-of-the-art detection. The comparison is matched on purpose. The PQC never sees pixels. Product and entangling circuits differ only by a CNOT ring (6,188 vs 6,215 trainable parameters). Forty-five of forty-five scheduled runs are reported, including seven degenerate temperature fits. Pneumonia is not claimed across datasets because VinDr-CXR has no Pneumonia class; Consolidation ∪ Infiltration is an ontology shift and is confined to the appendix. Temperature scaling is fitted on CheXpert validation logits and tested for transfer; it can lower source ECE and raise target ECE on the same run.

We do not claim quantum advantage, hardware execution, or clinical deployment. Review is single-blind; author name remains in the PDF. Citations follow the official `midl.cls` / natbib author–year style.

The work is original, not under consideration elsewhere, and uses only public de-identified datasets (CheXpert, VinDr-CXR). The author has no competing interests.

Sincerely,  
Tran Thi Thuy Quyen
