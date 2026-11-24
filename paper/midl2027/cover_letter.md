# MIDL 2027 Main Track — submission note

**Manuscript title:** What a Six-Qubit Head Can and Cannot Change: Calibration under Cross-Population Chest X-Ray Shift at Matched Capacity

**Track:** Main Track — full paper (PMLR), not short paper  
**Author:** Tran Thi Thuy Quyen  
Information Assurance, FPT University Can Tho, Vietnam  
quyentttiace200993@gmail.com

Dear Program Chairs,

Please consider this full paper for the MIDL 2027 Main Track. It reports a negative but controlled result: on a shared six-dimensional bottleneck from a frozen ImageNet DenseNet-121, a standard parameterized quantum circuit does not improve confidence calibration relative to a matched MLP when CheXpert-trained heads are evaluated on VinDr-CXR.

The manuscript does not propose a new architecture and does not claim state-of-the-art detection. The comparison is matched on purpose: the PQC never sees pixels and all heads are trained identically. A central contribution of this revision is a methodological one that is easy to overlook: with a single-qubit readout and no entangling gates, the product PQC is functionally a one-feature model, so nominal parameter parity does not imply functional parity. We state this openly and read the results accordingly (the clean classical-versus-quantum comparison is MLP versus entangling PQC). Forty-five of forty-five scheduled runs are reported, including seven degenerate temperature fits. Calibration is reported on both balanced and natural-prevalence VinDr holdouts, and the full metric set (AUROC, AUPRC, F1, Brier, ECE, MCE) is reported. Pneumonia is not claimed across datasets because VinDr-CXR has no Pneumonia class; Consolidation ∪ Infiltration is an ontology shift and is confined to the appendix. Temperature scaling is fitted on CheXpert validation logits and tested for transfer; it can lower source ECE and raise target ECE on the same run.

We do not claim quantum advantage, hardware execution, or clinical deployment. Review is single-blind; the author name remains in the PDF. Citations follow the official `midl.cls` / natbib author–year style.

The work is original, not under consideration elsewhere, and uses only public de-identified datasets (CheXpert, VinDr-CXR). The author has no competing interests.

Sincerely,  
Tran Thi Thuy Quyen