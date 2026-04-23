# ============================================================
#  CHAPTER 4 — BIAS DETECTION & ANALYSIS
# 
#  Dataset: German Credit Dataset
# ============================================================
# 
# Python 3.10+
# 


# ── STEP 1: Import Libraries ────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

print("=" * 55)
print("  CHAPTER 4: BIAS DETECTION & ANALYSIS")
print("=" * 55)


# ── STEP 2: Load Dataset ────────────────────────────────────

print(f"\n[INFO] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")


# ── STEP 3: Explore the Dataset ─────────────────────────────
print("\n--- SECTION 3.1: Dataset Overview ---")
print(f"Total applicants : {len(df)}")
print(f"Good credit risk : {(df['class'] == 1).sum()}  ({(df['class']==1).mean()*100:.1f}%)")
print(f"Bad credit risk  : {(df['class'] == 0).sum()}  ({(df['class']==0).mean()*100:.1f}%)")

print("\n--- Gender Distribution ---")
print(df['sex'].value_counts())

print("\n--- Age Group Distribution ---")
df['age_group'] = df['age'].apply(lambda x: 'Young (<25)' if x < 25 else 'Older (>=25)')
print(df['age_group'].value_counts())

print("\n--- Foreign Worker Distribution ---")
fw_labels = {'A201': 'Foreign', 'A202': 'Non-Foreign'}
df['fw_label'] = df['foreign_worker'].map(fw_labels)
print(df['fw_label'].value_counts())


# ── STEP 4: Compute Approval Rates by Group ─────────────────
print("\n--- SECTION 3.2: Approval Rates by Demographic Group ---")

# Gender
male_rate   = df[df['sex'] == 'male']['class'].mean()
female_rate = df[df['sex'] == 'female']['class'].mean()

# Age
young_rate  = df[df['age_group'] == 'Young (<25)']['class'].mean()
older_rate  = df[df['age_group'] == 'Older (>=25)']['class'].mean()

# Foreign worker
foreign_rate    = df[df['fw_label'] == 'Foreign']['class'].mean()
nonforeign_rate = df[df['fw_label'] == 'Non-Foreign']['class'].mean()

print(f"\nGender:")
print(f"  Male approval rate   : {male_rate:.3f} ({male_rate*100:.1f}%)")
print(f"  Female approval rate : {female_rate:.3f} ({female_rate*100:.1f}%)")

print(f"\nAge Group:")
print(f"  Young (<25) approval rate  : {young_rate:.3f} ({young_rate*100:.1f}%)")
print(f"  Older (>=25) approval rate : {older_rate:.3f} ({older_rate*100:.1f}%)")

print(f"\nForeign Worker Status:")
print(f"  Foreign approval rate     : {foreign_rate:.3f} ({foreign_rate*100:.1f}%)")
print(f"  Non-Foreign approval rate : {nonforeign_rate:.3f} ({nonforeign_rate*100:.1f}%)")


# ── STEP 5: Compute Bias Metrics (SPD and DIR) ───────────────
print("\n--- SECTION 3.3: Bias Metrics ---")

# Statistical Parity Difference (SPD) = unprivileged - privileged
# Disparate Impact Ratio (DIR)        = unprivileged / privileged
# Threshold for bias: |SPD| > 0.10 and DIR < 0.80

spd_sex     = female_rate - male_rate
dir_sex     = female_rate / male_rate

spd_age     = young_rate - older_rate
dir_age     = young_rate / older_rate

spd_foreign = foreign_rate - nonforeign_rate
dir_foreign = foreign_rate / nonforeign_rate

def bias_flag(spd, dir_val):
    flags = []
    if abs(spd) > 0.10:
        flags.append("SPD BIAS DETECTED")
    if dir_val < 0.80:
        flags.append("DIR BIAS DETECTED")
    return " | ".join(flags) if flags else "Within threshold"

print(f"\n{'Attribute':<20} {'SPD':>8} {'DIR':>8}  {'Assessment'}")
print("-" * 65)
print(f"{'Sex (F vs M)':<20} {spd_sex:>8.3f} {dir_sex:>8.3f}  {bias_flag(spd_sex, dir_sex)}")
print(f"{'Age (Y vs O)':<20} {spd_age:>8.3f} {dir_age:>8.3f}  {bias_flag(spd_age, dir_age)}")
print(f"{'Foreign Worker':<20} {spd_foreign:>8.3f} {dir_foreign:>8.3f}  {bias_flag(spd_foreign, dir_foreign)}")
print("\nThresholds: |SPD| > 0.10 = bias | DIR < 0.80 = bias")


# ── STEP 6: Train Logistic Regression Model ─────────────────
print("\n--- SECTION 3.4: Model Training ---")

# Prepare features (drop non-numeric helpers we added)
df_model = df.drop(columns=['age_group', 'fw_label'])

# One-hot encode categorical variables
X = pd.get_dummies(df_model.drop(columns=['class']), drop_first=True)
y = df_model['class']

# Train/test split (70/30, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# Standardize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
report   = classification_report(y_test, y_pred, output_dict=True)

print(f"\nModel Accuracy : {accuracy:.3f} ({accuracy*100:.1f}%)")
print(f"Precision (Good risk) : {report['1']['precision']:.3f}")
print(f"Recall    (Good risk) : {report['1']['recall']:.3f}")
print(f"F1-Score  (Good risk) : {report['1']['f1-score']:.3f}")

# Store for Chapter 5 comparison
baseline_accuracy = accuracy
print("\n[INFO] Baseline accuracy saved for Chapter 5 comparison.")


# ── STEP 7: Generate Charts ──────────────────────────────────
print("\n--- Generating Charts ---")

sns.set_theme(style="whitegrid", palette="muted")
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
fig.suptitle("Figure 4.1: Loan Approval Rates by Demographic Group\n(Before Mitigation)",
             fontsize=14, fontweight='bold', y=1.02)

# Chart 1: Gender
groups_sex  = ['Male', 'Female']
rates_sex   = [male_rate * 100, female_rate * 100]
colors_sex  = ['#4472C4', '#ED7D31']
bars = axes[0].bar(groups_sex, rates_sex, color=colors_sex, width=0.5, edgecolor='white', linewidth=1.5)
axes[0].set_title('By Gender', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Approval Rate (%)')
axes[0].set_ylim(0, 100)
for bar, rate in zip(bars, rates_sex):
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                 f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
axes[0].axhline(y=rates_sex[0], color='gray', linestyle='--', alpha=0.5, label='Male baseline')

# Chart 2: Age
groups_age = ['Young\n(<25)', 'Older\n(≥25)']
rates_age  = [young_rate * 100, older_rate * 100]
colors_age = ['#ED7D31', '#4472C4']
bars2 = axes[1].bar(groups_age, rates_age, color=colors_age, width=0.5, edgecolor='white', linewidth=1.5)
axes[1].set_title('By Age Group', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Approval Rate (%)')
axes[1].set_ylim(0, 100)
for bar, rate in zip(bars2, rates_age):
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                 f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')

# Chart 3: Foreign Worker
groups_fw = ['Foreign\nWorker', 'Non-Foreign\nWorker']
rates_fw  = [foreign_rate * 100, nonforeign_rate * 100]
colors_fw = ['#ED7D31', '#4472C4']
bars3 = axes[2].bar(groups_fw, rates_fw, color=colors_fw, width=0.5, edgecolor='white', linewidth=1.5)
axes[2].set_title('By Foreign Worker Status', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Approval Rate (%)')
axes[2].set_ylim(0, 100)
for bar, rate in zip(bars3, rates_fw):
    axes[2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                 f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('figure4_1_approval_rates.png', dpi=150, bbox_inches='tight')
plt.close()
print("[SAVED] figure4_1_approval_rates.png")

# Chart 2: Bias metrics summary
fig2, ax = plt.subplots(figsize=(9, 5))
attributes  = ['Sex\n(F vs M)', 'Age\n(Young vs Older)', 'Foreign Worker\n(F vs NF)']
spd_values  = [abs(spd_sex), abs(spd_age), abs(spd_foreign)]
bar_colors  = ['#C00000' if v > 0.10 else '#70AD47' for v in spd_values]
bars4 = ax.bar(attributes, spd_values, color=bar_colors, width=0.4, edgecolor='white', linewidth=1.5)
ax.axhline(y=0.10, color='red', linestyle='--', linewidth=2, label='Bias threshold (0.10)')
ax.set_title('Figure 4.2: Statistical Parity Difference by Protected Attribute\n(Red = Bias Detected)',
             fontsize=12, fontweight='bold')
ax.set_ylabel('|Statistical Parity Difference|')
ax.set_ylim(0, 0.30)
for bar, val in zip(bars4, spd_values):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
            f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('figure4_2_spd_chart.png', dpi=150, bbox_inches='tight')
plt.close()
print("[SAVED] figure4_2_spd_chart.png")

print("\n" + "=" * 55)
print("  CHAPTER 4 ANALYSIS COMPLETE")
print(f"  Baseline model accuracy: {baseline_accuracy*100:.1f}%")
print(f"  Gender SPD: {spd_sex:.3f}  DIR: {dir_sex:.3f}")
print(f"  Age SPD:    {spd_age:.3f}  DIR: {dir_age:.3f}")
print("  --> Run chapter5_mitigation.py for next step")
print("=" * 55)
