import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create output metrics directory
output_dir = "outputs/metrics"
os.makedirs(output_dir, exist_ok=True)

# 1. Define Pilot Metrics Data
epochs = [1, 2, 3, 4]
data = {
    "Epoch": [1, 2, 3, 4],
    "Generator_Loss": [16.3636, 15.0112, 14.8850, 14.9571],
    "Discriminator_Loss": [0.6115, 0.3421, 0.3892, 0.4374],
    "L1_Loss": [0.1451, 0.1284, 0.1265, 0.1271],
    "Adversarial_Loss": [0.9368, 1.2541, 1.3412, 1.4784],
    "SAM_Loss": [0.0916, 0.0820, 0.0782, 0.0765],
    "Val_PSNR_dB": [19.75, 20.12, 20.25, 20.37],
    "Val_SSIM": [0.3589, 0.3512, 0.3490, 0.3465],
    "Val_SAM_deg": [4.80, 5.12, 5.35, 5.64]
}

df = pd.DataFrame(data)

# Save pilot_metrics.csv
csv_path = os.path.join(output_dir, "pilot_metrics.csv")
df.to_csv(csv_path, index=False)
print(f"[Saved] CSV saved to: {csv_path}")

# 2. Percentage Change Summary Calculation
summary_data = []
for col in df.columns:
    if col == "Epoch":
        continue
    val_e1 = df[col].iloc[0]
    val_e4 = df[col].iloc[-1]
    pct_change = ((val_e4 - val_e1) / val_e1) * 100.0
    summary_data.append({
        "Metric": col,
        "Epoch_1_Value": val_e1,
        "Epoch_4_Value": val_e4,
        "Absolute_Change": val_e4 - val_e1,
        "Percentage_Change_%": round(pct_change, 3)
    })

df_summary = pd.DataFrame(summary_data)
summary_csv_path = os.path.join(output_dir, "metrics_summary.csv")
df_summary.to_csv(summary_csv_path, index=False)
print(f"[Saved] Summary CSV saved to: {summary_csv_path}")

# Save text summary
summary_txt_path = os.path.join(output_dir, "metrics_summary.txt")
with open(summary_txt_path, "w") as f:
    f.write("=== SIH26142 PILOT EXPERIMENT METRICS SUMMARY (Epoch 1 -> Epoch 4) ===\n\n")
    for _, row in df_summary.iterrows():
        sign = "+" if row['Percentage_Change_%'] >= 0 else ""
        f.write(f" • {row['Metric']:<20}: Epoch 1 = {row['Epoch_1_Value']:<8} | Epoch 4 = {row['Epoch_4_Value']:<8} | Change = {sign}{row['Percentage_Change_%']:.3f}%\n")

print(f"[Saved] Summary Text file saved to: {summary_txt_path}")

# 3. Plotting Styling Configuration
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 1.0

# Function for individual single-metric publication plots
def create_single_plot(x, y, ylabel, title, color, save_name):
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.plot(x, y, marker='o', linewidth=2.5, markersize=8, color=color, label=title)
    ax.set_xticks(x)
    ax.set_xlabel("Epoch", fontsize=12, fontweight='bold', labelpad=8)
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold', labelpad=8)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Annotate value points
    for i, txt in enumerate(y):
        ax.annotate(f"{txt:.4f}" if isinstance(txt, float) and txt < 10 else f"{txt:.2f}",
                    (x[i], y[i]), textcoords="offset points", xytext=(0, 8),
                    ha='center', fontsize=10, fontweight='bold', color='#333333')
        
    y_margin = (max(y) - min(y)) * 0.25 if max(y) != min(y) else 0.1
    ax.set_ylim(min(y) - y_margin, max(y) + y_margin)
    
    plt.tight_layout()
    out_file = os.path.join(output_dir, save_name)
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"[Plot Saved] {out_file}")

# Generate individual plots
create_single_plot(epochs, df["Generator_Loss"], "Generator Loss (Total)", "Generator Loss Progression", "#1f77b4", "generator_loss.png")
create_single_plot(epochs, df["Discriminator_Loss"], "Discriminator Loss (BCE)", "Discriminator Loss Progression", "#d62728", "discriminator_loss.png")
create_single_plot(epochs, df["L1_Loss"], "L1 Pixel Reconstruction Loss", "L1 Loss Progression", "#2ca02c", "l1_loss.png")
create_single_plot(epochs, df["Adversarial_Loss"], "Adversarial BCE Loss", "Adversarial Loss Progression", "#ff7f0e", "adversarial_loss.png")
create_single_plot(epochs, df["SAM_Loss"], "Spectral Angle Mapper Loss (rad)", "Train SAM Loss Progression", "#9467bd", "sam_loss.png")

# 4. Generate validation_metrics.png (3 Separate Subplot Panels)
fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)

# Panel 1: PSNR (dB)
axes[0].plot(epochs, df["Val_PSNR_dB"], marker='s', linewidth=2.5, markersize=8, color='#1f77b4')
axes[0].set_xticks(epochs)
axes[0].set_xlabel("Epoch", fontsize=11, fontweight='bold')
axes[0].set_ylabel("PSNR (dB)", fontsize=11, fontweight='bold')
axes[0].set_title("Validation PSNR (dB)", fontsize=13, fontweight='bold')
axes[0].grid(True, linestyle='--', alpha=0.6)
for i, txt in enumerate(df["Val_PSNR_dB"]):
    axes[0].annotate(f"{txt:.2f} dB", (epochs[i], txt), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9, fontweight='bold')
axes[0].set_ylim(19.5, 20.6)

# Panel 2: SSIM (0-1)
axes[1].plot(epochs, df["Val_SSIM"], marker='^', linewidth=2.5, markersize=8, color='#2ca02c')
axes[1].set_xticks(epochs)
axes[1].set_xlabel("Epoch", fontsize=11, fontweight='bold')
axes[1].set_ylabel("SSIM (Unitless: 0-1)", fontsize=11, fontweight='bold')
axes[1].set_title("Validation SSIM", fontsize=13, fontweight='bold')
axes[1].grid(True, linestyle='--', alpha=0.6)
for i, txt in enumerate(df["Val_SSIM"]):
    axes[1].annotate(f"{txt:.4f}", (epochs[i], txt), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9, fontweight='bold')
axes[1].set_ylim(0.34, 0.365)

# Panel 3: SAM Angle (Degrees)
axes[2].plot(epochs, df["Val_SAM_deg"], marker='D', linewidth=2.5, markersize=8, color='#d62728')
axes[2].set_xticks(epochs)
axes[2].set_xlabel("Epoch", fontsize=11, fontweight='bold')
axes[2].set_ylabel("SAM Spectral Angle (°)", fontsize=11, fontweight='bold')
axes[2].set_title("Validation SAM Angle (°)", fontsize=13, fontweight='bold')
axes[2].grid(True, linestyle='--', alpha=0.6)
for i, txt in enumerate(df["Val_SAM_deg"]):
    axes[2].annotate(f"{txt:.2f}°", (epochs[i], txt), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9, fontweight='bold')
axes[2].set_ylim(4.5, 6.0)

fig.suptitle("Validation Performance Evaluation Across Epochs 1–4", fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()

val_plot_path = os.path.join(output_dir, "validation_metrics.png")
plt.savefig(val_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"[Plot Saved] {val_plot_path}")

print("\n=== ALL METRICS AND PLOTS GENERATED SUCCESSFULLY ===")
