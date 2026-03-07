# Paper Notes

Running collection of ideas, observations, and reviewer-anticipation notes. Add freely; organize later.

---

## Supplemental Material Ideas

### S1: DDPM Comparison (2026-03-07)
**Include as supplemental figure + table.** Strong negative result that strengthens methodology.
- Best figure: DVH comparison (fig4) — predicted PTV curve shifted 60 Gy left of GT. Tells the whole story at a glance.
- Also consider: dose colorwash side-by-side showing diffuse low-magnitude prediction.
- Supplemental table: DDPM vs baseline vs combined loss on all metrics.
- **Narrative:** "We evaluated a conditional DDPM with equivalent architecture (~23.7M parameters, same U-Net backbone) but it failed to reconstruct target dose magnitudes, predicting PTV D95 of 4-6 Gy vs the 70 Gy target. This supports the choice of direct regression over generative modeling for VMAT dose prediction."
- Interpretation: dose prediction is a well-conditioned regression problem (anatomy + constraints → unique dose). Diffusion models add generative overhead without benefit.

### S2: Architecture Ablation
- Three architectures tested (Attention, BottleneckAttn, Wider), all comparable to baseline.
- Supplemental table with metrics. Brief mention in main text.
- Message: architecture is not the bottleneck for this problem size.

### S3: Loss Component Ablation
- Individual contribution of each loss term?
- Currently only have the combined results and the 3:1 / 2:1 / 2.5:1 tuning series.
- [ ] Consider a proper ablation: MSE only, +Gradient, +DVH, +Structure, +Asymmetric

### S4: Augmentation Ablation
- Augmentation vs no-augmentation results available (2026-02-27).
- Small effect — augmentation helps slightly.

---

## Anticipated Reviewer Questions

### "Why not use a larger dataset?"
- Acknowledge as limitation. Note that 200+ cases expected soon and will be included in revision if available.
- Emphasize that our approach achieves near-clinical metrics with only 74 cases — relevant for smaller centers.

### "How does this compare to [commercial product / other institution's results]?"
- Direct comparison is impossible (different datasets, disease sites, plan quality).
- Cite comparable metrics from literature where possible.
- Emphasize reproducibility: our code/pipeline is documented for replication.

### "Why not end-to-end (predict fluence, not dose)?"
- Out of scope — dose prediction is the accepted first step in knowledge-based planning.
- Fluence prediction is a separate research direction.

### "What about uncertainty quantification?"
- Current model produces point estimates only.
- Multi-seed variability provides some uncertainty measure.
- [ ] Consider MC dropout or ensemble uncertainty — future work section.

### "Global gamma is only 30% — isn't that bad?"
- Global gamma is dominated by low-dose voxels far from PTV where small absolute errors cause large relative failures.
- PTV-region gamma (94.3%) is the clinically relevant metric.
- Discuss whether global gamma is an appropriate metric for dose prediction.

---

## Narrative Framing Ideas

### "Loss engineering > architecture engineering"
Core message: with a fixed, well-established architecture (3D U-Net), the key to clinical metric optimization is the loss function, not the model.

### "Approaching clinical readiness with modest data"
74 cases → PTV gamma 94.3%, D95 gap 0.06 Gy. With dataset expansion, this could cross the 95% clinical threshold.

### "The asymmetric insight"
Symmetric loss functions (MSE) produce symmetric errors around the target. But clinical consequences are asymmetric — underdosing the tumor is far worse than mild overdose. The asymmetric PTV penalty encodes this clinical knowledge directly into the loss.

---

## Figure Design Notes

- Use consistent color scheme across all figures (Wong 2011 palette — already in PLOT_CONFIG)
- Main text figures should be self-contained (no need to flip to methods to understand)
- Consider a "graphical abstract" for the journal — dose colorwash or DVH comparison
