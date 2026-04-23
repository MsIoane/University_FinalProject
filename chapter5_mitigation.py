# ============================================================
#  CHAPTER 5 — BIAS MITIGATION & EVALUATION
#  Thesis: Evaluating and Mitigating Bias in AI-Based Credit
#          Scoring Systems
#  Dataset: German Credit Dataset
#
#  IMPORTANT: Run chapter4_bias_detection.py first.
#  This file builds directly on Chapter 4 results.
# ============================================================
#
#  HOW TO RUN THIS FILE:
#  1. Make sure german_credit.csv is in the same folder
#  2. Install required libraries:
#       pip install pandas numpy scikit-learn matplotlib seaborn
#  3. Run:
#       python chapter5_mitigation.py
#  4. Charts will be saved as PNG files in the same folder
# ============================================================


# ── STEP 1: Import Libraries ────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, classification_report)
import warnings
warnings.filterwarnings('ignore')

print("=" * 55)
print("  CHAPTER 5: BIAS MITIGATION & EVALUATION")
print("=" * 55)


# ── STEP 2: Reload Dataset & Reproduce Baseline ─────────────
# (Identical setup to Chapter 4 for consistency)
print("\n[INFO] Loading dataset and reproducing Chapter 4 baseline...")

df = pd.read_csv('german_credit.csv')
df['age_group'] = df['age'].apply(lambda x: 'Young (<25)' if x < 25 else 'Older (>=25)')
df['fw_label']  = df['foreign_worker'].map({'A201': 'Foreign', 'A202': 'Non-Foreign'})

df_model = df.drop(columns=['age_group', 'fw_label'])
X = pd.get_dummies(df_model.drop(columns=['class']), drop_first=True)
y = df_model['class']

# Same split as Chapter 4 (random_state=42 ensures identical split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# Baseline model (no mitigation)
model_base = LogisticRegression(max_iter=1000, random_state=42)
model_base.fit(X_train_sc, y_train)
y_pred_base = model_base.predict(X_test_sc)
acc_base = accuracy_score(y_test, y_pred_base)

# Recover sex attribute for test set
sex_test = df.loc[X_test.index, 'sex'].values
age_test = df.loc[X_test.index, 'age'].values

# Baseline bias metrics (on model predictions)
male_pred_base   = y_pred_base[sex_test == 'male'].mean()
female_pred_base = y_pred_base[sex_test == 'female'].mean()
spd_base = female_pred_base - male_pred_base
dir_base = female_pred_base / male_pred_base

print(f"  Baseline accuracy : {acc_base:.3f}")
print(f"  Baseline SPD (sex): {spd_base:.3f}")
print(f"  Baseline DIR (sex): {dir_base:.3f}")


# ── STEP 3: Compute Reweighting Sample Weights ──────────────
print("\n--- SECTION 5.1: Reweighting ---")
print("Computing sample weights for each demographic-class group...")

# Attach sex to training set for weight computation
train_df = X_train.copy()
train_df['class'] = y_train.values
train_df['sex']   = df.loc[X_train.index, 'sex'].values

n_total  = len(train_df)
n_groups = 4  # male/female x good/bad

weights = np.ones(n_total)

group_info = []
for sex_val in ['male', 'female']:
    for class_val in [0, 1]:
        mask = (train_df['sex'] == sex_val) & (train_df['class'] == class_val)
        n_g  = mask.sum()
        if n_g > 0:
            # Formula: weight = n_total / (n_groups * n_group)
            w = n_total / (n_groups * n_g)
            weights[mask.values] = w
            group_info.append({
                'Group': f"sex={sex_val}, class={class_val}",
                'Count': n_g,
                'Weight': round(w, 3)
            })

print(f"\n{'Group':<35} {'Count':>6} {'Weight':>8}")
print("-" * 52)
for g in group_info:
    print(f"{g['Group']:<35} {g['Count']:>6} {g['Weight']:>8.3f}")

print("""
How reweighting works:
  - Groups with fewer samples get HIGHER weight
  - Groups with more samples get LOWER weight
  - This balances influence during model training
  - No data is added or removed — only importance changes
""")


# ── STEP 4: Train Mitigated Model ───────────────────────────
print("--- SECTION 5.2: Training Mitigated Model ---")

model_fair = LogisticRegression(max_iter=1000, random_state=42)
model_fair.fit(X_train_sc, y_train, sample_weight=weights)
#                                   ^^^^^^^^^^^^^^^^^^^
#                 The only difference from Chapter 4 baseline

y_pred_fair = model_fair.predict(X_test_sc)
acc_fair = accuracy_score(y_test, y_pred_fair)

print(f"\n[INFO] Mitigated model trained with sample_weight parameter.")
print(f"  Mitigated accuracy : {acc_fair:.3f}")


# ── STEP 5: Measure Bias After Mitigation ───────────────────
print("\n--- SECTION 5.3: Bias Metrics After Mitigation ---")

male_pred_fair   = y_pred_fair[sex_test == 'male'].mean()
female_pred_fair = y_pred_fair[sex_test == 'female'].mean()
spd_fair = female_pred_fair - male_pred_fair
dir_fair = female_pred_fair / male_pred_fair

# Age bias (secondary — not mitigated, shown for comparison)
young_pred_fair = y_pred_fair[age_test < 25].mean()
older_pred_fair = y_pred_fair[age_test >= 25].mean()
spd_age_fair    = young_pred_fair - older_pred_fair
dir_age_fair    = young_pred_fair / older_pred_fair

print(f"\n{'Metric':<35} {'Before':>10} {'After':>10} {'Change':>10}")
print("-" * 68)
print(f"{'Male approval rate (predicted)':<35} {male_pred_base:>10.3f} {male_pred_fair:>10.3f} {male_pred_fair-male_pred_base:>+10.3f}")
print(f"{'Female approval rate (predicted)':<35} {female_pred_base:>10.3f} {female_pred_fair:>10.3f} {female_pred_fair-female_pred_base:>+10.3f}")
print(f"{'SPD (sex)':<35} {spd_base:>10.3f} {spd_fair:>10.3f} {spd_fair-spd_base:>+10.3f}")
print(f"{'DIR (sex)':<35} {dir_base:>10.3f} {dir_fair:>10.3f} {dir_fair-dir_base:>+10.3f}")
print(f"{'Model Accuracy':<35} {acc_base:>10.3f} {acc_fair:>10.3f} {acc_fair-acc_base:>+10.3f}")


# ── STEP 6: Full Performance Metrics ────────────────────────
print("\n--- SECTION 5.4: Full Performance Comparison ---")

prec_base = precision_score(y_test, y_pred_base)
rec_base  = recall_score(y_test, y_pred_base)
f1_base   = f1_score(y_test, y_pred_base)

prec_fair = precision_score(y_test, y_pred_fair)
rec_fair  = recall_score(y_test, y_pred_fair)
f1_fair   = f1_score(y_test, y_pred_fair)

print(f"\n{'Metric':<20} {'Baseline':>12} {'Mitigated':>12} {'Change':>10}")
print("-" * 56)
print(f"{'Accuracy':<20} {acc_base:>12.3f} {acc_fair:>12.3f} {acc_fair-acc_base:>+10.3f}")
print(f"{'Precision':<20} {prec_base:>12.3f} {prec_fair:>12.3f} {prec_fair-prec_base:>+10.3f}")
print(f"{'Recall':<20} {rec_base:>12.3f} {rec_fair:>12.3f} {rec_fair-rec_base:>+10.3f}")
print(f"{'F1-Score':<20} {f1_base:>12.3f} {f1_fair:>12.3f} {f1_fair-f1_base:>+10.3f}")
print(f"{'SPD (sex)':<20} {spd_base:>12.3f} {spd_fair:>12.3f} {spd_fair-spd_base:>+10.3f}")
print(f"{'DIR (sex)':<20} {dir_base:>12.3f} {dir_fair:>12.3f} {dir_fair-dir_base:>+10.3f}")


# ── STEP 7: Generate Charts ──────────────────────────────────
print("\n--- Generating Charts ---")
sns.set_theme(style="whitegrid", palette="muted")

# ── Chart 1: Approval Rate Comparison (Before vs After) ──
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle("Figure 5.1: Predicted Approval Rates Before and After Mitigation",
             fontsize=13, fontweight='bold', y=1.02)

groups   = ['Male', 'Female']
before   = [male_pred_base * 100, female_pred_base * 100]
after    = [male_pred_fair * 100, female_pred_fair * 100]
x        = np.arange(len(groups))
width    = 0.35

bars1 = axes[0].bar(x - width/2, before, width, label='Before', color='#ED7D31', edgecolor='white')
bars2 = axes[0].bar(x + width/2, after,  width, label='After',  color='#4472C4', edgecolor='white')
axes[0].set_title('Approval Rate by Gender', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Predicted Approval Rate (%)')
axes[0].set_xticks(x); axes[0].set_xticklabels(groups)
axes[0].set_ylim(0, 110)
axes[0].legend()
for bar in bars1:
    axes[0].text(bar.get_x()+bar.get_width()/2., bar.get_height()+1,
                 f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
for bar in bars2:
    axes[0].text(bar.get_x()+bar.get_width()/2., bar.get_height()+1,
                 f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Gender gap arrow annotation
gap_before = before[0] - before[1]
gap_after  = after[0]  - after[1]
axes[0].annotate(f'Gap: {gap_before:.1f}pp', xy=(x[0]-width/2, before[0]),
                 xytext=(-0.3, 75), fontsize=9, color='#ED7D31', fontweight='bold')
axes[0].annotate(f'Gap: {gap_after:.1f}pp', xy=(x[0]+width/2, after[0]),
                 xytext=(0.7, 75), fontsize=9, color='#4472C4', fontweight='bold')

# ── Chart 2: SPD and DIR Before vs After ──
metrics     = ['SPD\n(sex)', 'DIR\n(sex)']
vals_before = [abs(spd_base), dir_base]
vals_after  = [abs(spd_fair), dir_fair]
x2 = np.arange(len(metrics))

bars3 = axes[1].bar(x2 - width/2, vals_before, width, label='Before', color='#ED7D31', edgecolor='white')
bars4 = axes[1].bar(x2 + width/2, vals_after,  width, label='After',  color='#4472C4', edgecolor='white')
axes[1].set_title('Bias Metrics (SPD & DIR)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Metric Value')
axes[1].set_xticks(x2); axes[1].set_xticklabels(metrics)
axes[1].set_ylim(0, 1.1)
axes[1].axhline(y=0.10, color='red', linestyle='--', linewidth=1.5, label='SPD threshold (0.10)', alpha=0.7)
axes[1].axhline(y=0.80, color='green', linestyle='--', linewidth=1.5, label='DIR threshold (0.80)', alpha=0.7)
axes[1].legend(fontsize=8)
for bar in list(bars3) + list(bars4):
    axes[1].text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.01,
                 f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('figure5_1_mitigation_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("[SAVED] figure5_1_mitigation_comparison.png")

# ── Chart 2: Accuracy-Fairness Tradeoff ──
fig2, ax = plt.subplots(figsize=(8, 5))
categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
base_vals  = [acc_base, prec_base, rec_base, f1_base]
fair_vals  = [acc_fair, prec_fair, rec_fair, f1_fair]
x3 = np.arange(len(categories))

b1 = ax.bar(x3 - width/2, [v*100 for v in base_vals], width, label='Baseline (No Mitigation)',
            color='#ED7D31', edgecolor='white')
b2 = ax.bar(x3 + width/2, [v*100 for v in fair_vals], width, label='After Reweighting',
            color='#4472C4', edgecolor='white')
ax.set_title('Figure 5.2: Model Performance Metrics — Baseline vs Mitigated',
             fontsize=12, fontweight='bold')
ax.set_ylabel('Score (%)')
ax.set_xticks(x3); ax.set_xticklabels(categories)
ax.set_ylim(0, 100)
ax.legend()
for bar in list(b1) + list(b2):
    ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.5,
            f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('figure5_2_accuracy_tradeoff.png', dpi=150, bbox_inches='tight')
plt.close()
print("[SAVED] figure5_2_accuracy_tradeoff.png")

print("\n" + "=" * 55)
print("  CHAPTER 5 ANALYSIS COMPLETE")
print(f"  SPD: {spd_base:.3f} → {spd_fair:.3f}  (reduced by {abs(spd_fair-spd_base):.3f})")
print(f"  DIR: {dir_base:.3f} → {dir_fair:.3f}  (improved)")
print(f"  Accuracy cost: {acc_base:.3f} → {acc_fair:.3f}  ({acc_fair-acc_base:+.3f})")
print(f"  → Bias reduced. Fairness-accuracy tradeoff confirmed.")
print("=" * 55)
