# 🔬 Deep Research: Designing a Visual Debugger for Batched Neural Network Training

## 🧭 Overview
You're building a **visual debugger for neural networks** that offers 100% traceability per training step. In `PER_SAMPLE` mode, the pipeline is elegantly simple:

```
Predict ➔ Compute Blame (Error Signal) ➔ Adjust Weights
```

Each weight update is shown in a popup that details:
- Inputs
- My Responsibility ("Blame")
- Learning Rate
- Adjustment Calculation
- Before & After values

✅ This flow is **intuitive, educational, and auditable.**

---

## 🚧 Challenge: Introducing Batching Without Losing Auditability
In **batch-based training**, weight updates are deferred. Instead of applying updates immediately per sample, you accumulate gradients and apply a **single update per batch**.

Examples:
- Batch size = 1 → updates per sample
- Batch size = 4 → update every 4 samples
- Batch size = dataset size → update once per epoch

This **breaks the clean per-sample adjustment audit trail**.

### ❌ Problem:
When the weight updates are applied later (after multiple samples), the popup loses its real-time traceability.

---

## 🔥 Non-Negotiable Requirement
You must preserve **100% traceability** of weight updates.

Every final weight adjustment must be explainable by showing **exactly how each sample contributed**.

---

## 🎯 Goal
Redesign the **Adjust Weights popup** (and training step visualization) to:

- ✅ Provide full traceability even with deferred updates.
- ✅ Maintain simplicity and clarity.
- ✅ Stay true to the mental model:

```
Predict ➔ Blame ➔ Adjust
```

---

## 🧠 Visualization and UX Strategies

### 🔹 1. Split View: Backpass → Accumulate + Adjust
- Keep one row per weight.
- Split the **Backpass column** into:
  - Accumulation (shown per sample)
  - Adjustment (same across all samples in batch)
- Use color to distinguish accumulation vs final adjustment (e.g., yellow vs green).

### 🔹 2. Popup Logic: Mirror Existing View, Add Adjustment Section
- The popup stays the same.
- **Only difference:** Replace "adjustment" with **accumulation** per sample.
- Add a new panel below/right to show the **final actual update**.
- This final adjustment panel is **identical across all samples in batch**.

### 🔹 3. Adjustment Once Per Epoch/Batch
- You could choose to:
  - Show final adjustment **on the last sample only** (saves space, clean)
  - Or show it on **every sample**, marked as final once it's committed

### 🔹 4. Keep Layout Constant
- Don’t shift UI elements.
- Don’t collapse or expand rows.
- Instead, color and label the difference in behavior (Pending vs Final).

---

## 🧾 Naming and Terminology
| Concept               | Suggested Label             |
|----------------------|-----------------------------|
| Accumulated Gradients| Blame Bank / Accumulator    |
| Final Adjustment     | Batch Commit / Vault Apply  |
| Deferred state       | Pending Update              |
| Sample contribution  | Blame Slice / Responsibility Share |
| View name            | Adjustment Trace / Epoch Summary |

---

## ✅ Summary: What Stays, What Changes
### ✅ What stays the same:
- Per-sample forward pass
- Per-sample error (blame)
- Display of inputs and blame per sample

### ⚠️ What changes:
- Weight update is **delayed** until end of batch
- Popup must now show **accumulation**, not adjustment
- A new section must show **final adjustment** (same for all samples in batch)

---

## 🚀 TL;DR for Implementation
> **Per sample:** Show "blame slice" + accumulated state
>
> **Once per batch:** Show "final adjustment" to weights

Color-coded, label-rich, layout-consistent.

✅ 100% auditable.
✅ Consistent with mental model.
✅ Still beautiful, simple, and powerful.

---

## 🧪 Bonus Ideas
- Add hover to show "how much each sample contributed"
- Mini bar graph: growing contribution per sample
- Sidebar panel: full epoch summary

---

## Notes of highlights

- Piggy Bank Metaphor
## Implementation
- 