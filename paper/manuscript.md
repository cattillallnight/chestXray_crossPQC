# Matched-Capacity Hybrid Quantum-Classical Networks for Confidence Calibration under Cross-Population Chest X-Ray Shift

Tran Thi Thuy Quyen  
Information Assurance, FPT University Can Tho  
alwayshavebeenyaa@gmail.com

---

## Abstract

Hybrid quantum-classical networks are often presented as drop-in replacements for convolutional classifiers. That comparison is not well posed when the quantum circuit sees a handful of engineered scalars while the classical baseline sees the full image. This paper asks a narrower question that the implementation can actually answer. On a shared six-dimensional bottleneck taken from a frozen ImageNet DenseNet-121, does a parameterized quantum circuit (PQC) change discrimination or confidence calibration relative to a capacity-matched multilayer perceptron (MLP) when the label distribution shifts from CheXpert (United States) to VinDr-CXR (Vietnam)? Three heads are trained: an MLP, a product PQC without entangling gates, and an entangling PQC with a CNOT ring. All heads map six angle-embedded features to a single logit; the PQC measures \(\langle Z\rangle\) on qubit 0. Forty-five runs (five tasks, three heads, three seeds) are reported. On CheXpert test, PQC heads do not outperform the MLP on AUROC. Entanglement does not systematically reduce expected calibration error (ECE). Temperature scaling fitted on CheXpert validation logits can lower source ECE and raise target ECE on the same run. VinDr-CXR has no Pneumonia class; the pneumonia proxy (Consolidation \(\cup\) Infiltration) is an ontology shift and is confined to the appendix. The empirical boundary is the result: at this capacity, on this simulator, a PQC is not a more trustworthy head under cross-population shift.

**Keywords:** confidence calibration; domain shift; chest radiography; parameterized quantum circuit; matched capacity

## 1. Introduction

Chest radiograph classifiers that look well calibrated on a source hospital can become over- or under-confident on another population (Ovadia et al., 2019; Zech et al., 2018). Calibration under shift is therefore a clinical property, not a training curiosity. In parallel, variational quantum circuits have been attached to classical backbones and described as hybrid quantum-classical networks (Benedetti et al., 2019; Cerezo et al., 2021). Many of those studies compare a small PQC with a fully trained convolutional classifier, or they collapse label ontologies across datasets in order to claim a cross-site pneumonia detector. Both moves inflate the apparent role of the circuit.

The model studied here, Q-CalibNet, is deliberately smaller than that rhetoric. A DenseNet-121 (Huang et al., 2017) pretrained on ImageNet is frozen and used only as a 1024-dimensional feature extractor. A linear map compresses those features to six tanh-scaled angles. Three heads of nearly equal parameter count then produce a binary logit: a two-layer MLP, a product PQC, and an entangling PQC. The research questions follow the code.

- **RQ1 (secondary).** On the same six-dimensional bottleneck, do product or entangling PQCs differ from the MLP in AUROC, AUPRC, or F1 when models trained on CheXpert are evaluated on VinDr-CXR?
- **RQ2 (primary).** Does entanglement reduce ECE, maximum calibration error (MCE), Brier score, or overconfidence under shift, relative to a product PQC and the MLP?
- **RQ3.** Does a scalar temperature fitted on CheXpert validation logits calibrate PQC heads more effectively than the MLP on the Vietnamese target?

The contribution is the protocol, not a supremacy claim. Capacity is matched. Product versus entangling circuits isolate the inductive bias of CNOT. CheXpert uncertain labels are dropped (U-Ignore). VinDr multi-radiologist boxes are reduced with an any-vote. Tasks with a shared finding name are reported in the main tables; pneumonia is not.

## 2. Related work

### 2.1 Confidence calibration

A probabilistic classifier is calibrated when, among instances predicted with confidence \(p\), the empirical frequency of the predicted event is also \(p\). Brier (1950) introduced a proper scoring rule that jointly penalizes discrimination and calibration. Naeini, Cooper, and Hauskrecht (2015) defined expected calibration error by binning confidence and averaging the absolute gap between accuracy and confidence; Nixon et al. (2019) later documented the sensitivity of that estimator to binning. Platt (1999) and Zadrozny and Elkan (2002) treated calibration as a post-hoc map from scores to probabilities. Guo, Pleiss, Sun, and Weinberger (2017) showed that modern deep networks are systematically overconfident and that a single temperature \(T\) fitted by negative log-likelihood on a validation set—temperature scaling—is a strong, almost parameter-free baseline. Kull et al. (2019) generalized the idea to Dirichlet calibration in the multi-class setting. None of these methods assumes that the validation distribution matches the deployment distribution. Ovadia et al. (2019) showed that predictive uncertainty, including temperature-scaled confidence, degrades under covariate and semantic shift. The present study uses Guo-style scalar temperature as RQ3, fitted only on CheXpert validation logits, and tests whether that \(T\) transfers to VinDr-CXR.

### 2.2 Chest radiography, labels, and shift

Large CXR corpora made convolutional screening feasible and also made dataset shift visible. ChestX-ray8 (Wang et al., 2017), CheXpert (Irvin et al., 2019), MIMIC-CXR (Johnson et al., 2019), and VinDr-CXR (Nguyen et al., 2022) differ in geography, devices, report style, and label generation. CheXNet (Rajpurkar et al., 2017) reported radiologist-level pneumonia detection on a single corpus; subsequent analyses showed that such numbers need not travel. Zech et al. (2018) documented hospital-specific shortcuts in pneumonia CNNs. Oakden-Rayner et al. (2020) described hidden stratification: a high AUROC can hide failure on clinically important subtypes. Seyyed-Kalantari et al. (2021) showed underdiagnosis bias in underserved groups. Pooch, Ballester, and Barros (2020) measured domain shift among CXR classifiers directly. CheXpert encodes uncertainty as \(-1\) and defines U-Ignore, U-Ones, and U-Zeros; collapsing those conventions, or mapping CheXpert Pneumonia onto VinDr classes that do not include Pneumonia, is an ontology change rather than a population change. VinDr-CXR provides box annotations from multiple radiologists on 15,000 training radiographs collected in Vietnam. The present protocol keeps U-Ignore, uses an any-vote on VinDr boxes, reports four overlapping findings in the main table, and isolates Consolidation \(\cup\) Infiltration as an appendix proxy.

### 2.3 Parameterized quantum circuits as models

Parameterized quantum circuits are variational models whose gates carry trainable angles (Mitarai et al., 2018; Benedetti et al., 2019; Cerezo et al., 2021). Angle embedding of a classical vector, followed by layers of single-qubit rotations and optional entangling gates, is the default encoding in several software stacks, including PennyLane (Bergholm et al., 2018). Schuld and Killoran (2019) and Havlíček et al. (2019) framed such circuits as feature maps into Hilbert space. Schuld, Sweke, and Meyer (2021) showed that the encoding—not only the variational block—controls the Fourier spectrum of the hypothesis class. Pérez-Salinas et al. (2020) showed that re-uploading data can make even a single qubit a universal classifier in theory. Expressibility and entangling capability of common ansatze were quantified by Sim, Johnson, and Aspuru-Guzik (2019). Those capacity measures do not imply better calibration. Training is further constrained by barren plateaus (McClean et al., 2018). Huang et al. (2021) and Abbas et al. (2021) argued that quantum models can be powerful, but that classical data and geometric differences, not circuit depth alone, decide advantage. The implication for CXR work is prosaic: a six-qubit expectation head on a frozen DenseNet vector is a small classical-to-quantum map. Comparing it with a fully trained DenseNet classifier confounds representation learning with the head. Comparing a product circuit with an otherwise identical CNOT-ring circuit does not.

### 2.4 What this paper does not claim

This paper does not claim quantum advantage, hardware execution, or state-of-the-art CXR detection. It does not treat the unmatched DenseNet classifier as a fair baseline. It does not merge pneumonia across ontologies in the primary table. Related hybrid-imaging papers that do those things are not used as performance targets.

## 3. Methods

### 3.1 Source and target

CheXpert-v1.0-small is the source. Only frontal radiographs are kept. Uncertain labels (\(-1\)) are dropped; blank findings are treated as negative. Splits are patient-level. Training uses 2,000 class-balanced images; validation and test use 500 images at natural prevalence.

VinDr-CXR / VinBigData training images (15,000 PNG files) are the external target. Labels come from the public box CSV (`image_id`, `class_name`, `rad_id`). An image is positive for a finding if any radiologist assigned that class. No VinDr image is used for training. Two holdouts of 1,000 images are built: a class-balanced set and a natural-prevalence set. Consolidation has only 353 positives in the entire VinDr training set, so the “balanced” holdout is 353/647 rather than 500/500. The 3,000-image Kaggle test split has no public labels and is unused.

### 3.2 Tasks

Four primary tasks share a finding name on both datasets: no finding, cardiomegaly, pleural effusion, and consolidation. No finding is the cleanest ontology (positive = absence of any recorded abnormality). A fifth task, `pneumonia_proxy`, maps CheXpert Pneumonia to VinDr Consolidation \(\cup\) Infiltration and is flagged `ontology_shift=True`. It is not a pneumonia detector and is not used to claim cross-dataset pneumonia performance.

### 3.3 Architecture

Let \(x \in \mathbb{R}^{1024}\) be the frozen DenseNet-121 feature. The bottleneck is
\[
z = \pi \tanh(W x + b) \in \mathbb{R}^{6}.
\]
The MLP head is \(6 \to 8 \to \tanh \to \mathrm{Dropout}(0.1) \to 1\) (6,215 trainable parameters, including the bottleneck). Each PQC uses six wires, two layers of RX/RY/RZ, and Y-angle embedding of \(z\). The product circuit has no CNOTs; the entangling circuit adds a CNOT ring. The readout is \(\mathrm{scale}\cdot\langle Z_0\rangle + \mathrm{bias}\) (6,188 parameters). The bottleneck tensor is not detached: gradients flow into \(W\) for every head, which is the condition under which the comparison remains matched-capacity. Circuits are simulated with PennyLane `default.qubit` and backpropagation on CPU float64; there is no hardware run and no claim that six qubits process a \(1024\times 1024\) radiograph.

### 3.4 Training and calibration

Heads are trained with Adam (\(\mathrm{lr}=10^{-3}\), weight decay \(10^{-4}\)), binary cross-entropy with logits, 40 epochs, and early stopping of 8 epochs on validation AUROC. MLP batches are size 64 on CUDA; PQC batches are size 16 on CPU. Seeds are 42, 43, and 44. After training, a scalar temperature \(T\) is fitted by LBFGS on CheXpert validation logits, with \(T=\exp(\theta)\) clamped to \([10^{-3}, 100]\). Calibrated probabilities use \(\sigma(\mathrm{logit}/T)\). Metrics: AUROC, AUPRC, F1 at 0.5, Brier, 15-bin ECE and MCE. JSON records do not store logits; reliability diagrams are recomputed from checkpoints. Seven runs hit the temperature clamp; they stay in the main table and are removed only in a sensitivity ablation.

### 3.5 What is not compared

A fully trained DenseNet classifier is an unmatched upper bound. Fine-tuning the backbone, adding qubits mid-grid, merging pneumonia with consolidation in the main table, or describing PQC AUROC as quantum advantage would break the protocol.

## 4. Results

Forty-five of forty-five scheduled runs finished. VinDr metrics are present for every run. Mean \(\pm\) standard deviation is over three seeds. Degenerate temperatures (clamp \(10^{-3}\) or \(100\)) occurred on seven runs and are retained, not dropped; they are a property of the LBFGS fit on poorly scaled logits, especially under class imbalance.

### 4.1 RQ1 — discrimination

On CheXpert test, the MLP is at least as discriminative as either PQC on every primary task (no finding \(0.837\pm 0.003\) versus product \(0.820\pm 0.007\) and entangling \(0.813\pm 0.010\)). Under VinDr balanced evaluation, AUROC falls for no finding (MLP \(0.736\pm 0.008\), product \(0.669\pm 0.003\), entangling \(0.725\pm 0.015\)). Product PQC is not uniformly worse on the target: on consolidation its VinDr AUROC is \(0.714\pm 0.012\) against MLP \(0.682\pm 0.035\), inside seed noise and below any claim of advantage. Cardiomegaly product seed 43 is an outlier (CheXpert AUROC \(0.596\)). Across 12 primary task–seed pairs, product AUROC is below the matched MLP on 9 CheXpert tests; entangling AUROC is below the MLP on 10. Three seeds do not support a hierarchical significance test; the pattern is reported as paired mean differences, not as \(p\)-values.

### 4.2 RQ2 — entanglement and calibration

Entangling circuits do not yield lower source ECE. For no finding, CheXpert test ECE is \(0.320\pm 0.069\) (MLP), \(0.239\pm 0.031\) (product), and \(0.379\pm 0.038\) (entangling). The entangling head on no finding seed 42 is the clearest single-run illustration: test accuracy \(0.56\), sensitivity \(0.95\), ECE \(0.419\). Ring CNOTs at two layers and a single-qubit readout are not a calibration regularizer in this regime.

### 4.3 RQ3 — temperature transfer

Temperature fitted on CheXpert validation is not a reliable target calibrator. On no finding product PQC, mean CheXpert ECE falls after scaling (\(0.239\to 0.153\)) while VinDr balanced ECE rises (\(0.124\to 0.203\)). Seed 43 of that head is extreme: \(T=0.36\) reduces CheXpert ECE from \(0.212\) to \(0.054\) and increases VinDr balanced ECE from \(0.123\) to \(0.259\). Several other runs hit the clamp (\(T=100\) or \(T=0.001\)); those fits flatten or sharpen probabilities until ECE on one domain is an artifact of the box constraint. The honest statement of RQ3 is negative: source-fitted \(T\) does not systematically help PQC more than MLP on VinDr-CXR. This is consistent with Ovadia et al. (2019): a validation-fitted uncertainty map need not travel.

### 4.4 Appendix task

`pneumonia_proxy` CheXpert test prevalence is \(0.024\) (12 positives in 500). AUROC numbers on that split are high-variance and are not pneumonia performance. They are reported only to show that an ontology collapse would have been easy to overclaim.

## 5. Limitations

The circuit has six qubits, two layers, and a single observable. The backbone is frozen, so the PQC never sees pixels. Simulation uses `default.qubit`, not hardware noise. Sample sizes are small (train 2,000; val/test 500; VinDr 1,000). Three seeds cannot support a hierarchical significance test. Temperature is a scalar with a hard clamp. ECE uses 15 equal-width bins; rare positives make several bins empty. VinDr any-vote ignores radiologist disagreement. ImageNet-normalized \(224\times 224\) RGB conversion is a crude CXR preprocessing. None of these limitations is hidden in order to protect a quantum narrative.

## 6. Conclusion

Matched-capacity hybrid heads on a six-dimensional CXR bottleneck do not produce a more trustworthy classifier under CheXpert \(\to\) VinDr-CXR shift. Discrimination is comparable or slightly worse for PQC than for an MLP. Entanglement does not reduce ECE. Temperature scaling fitted on the source can harm the target. Those are reportable empirical boundaries. They are not evidence against quantum machine learning in general; they are evidence against overclaiming a six-qubit expectation head as a cross-population calibrator.

## Reproducibility

LaTeX: `paper/main.tex`, `paper/references.bib`. Interpreter: Python 3.10, `torch 2.6.0+cu124`, PennyLane 0.38.0. Grid: `scripts/run_queue.py`. Tables: `results/tables/all_runs.csv`, `paper_primary_mean_std.csv`. Figures: `results/figures/`. Checkpoints: `results/runs/*_seed*.pt`. Splits were not resampled after the grid.

## Ethics note

CheXpert and VinDr-CXR are public de-identified research datasets. No clinical deployment is claimed.
