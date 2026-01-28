import numpy as np
import matplotlib.pyplot as plt
import os

data = np.load("./AT_TD/epgilon_compare_results11/results_data.npz")
output_folder = './AT_TD/epgilon_compare_results12'
os.makedirs(output_folder, exist_ok=True)

epsilons = data["epsilons"]
ours_mean, ours_std   = data["ours_mean"],  data["ours_std"]
oursF_mean, oursF_std = data["oursF_mean"], data["oursF_std"]
ATR_mean, ATR_std     = data["ATR_mean"],   data["ATR_std"]
TRD_mean, TRD_std     = data["TRD_mean"],   data["TRD_std"]

# -------- Final Loss vs Epsilon --------
plt.figure(figsize=(10, 6))
plt.plot(epsilons, ours_mean, marker='o', label='Ours')
plt.fill_between(epsilons, ours_mean - ours_std, ours_mean + ours_std, alpha=0.2)

plt.plot(epsilons, oursF_mean, marker='o', label='Ours_F')
plt.fill_between(epsilons, oursF_mean - oursF_std, oursF_mean + oursF_std, alpha=0.2)

plt.plot(epsilons, ATR_mean, marker='o', label='Comparison')
plt.fill_between(epsilons, ATR_mean - ATR_std, ATR_mean + ATR_std, alpha=0.2)

plt.axhline(y=TRD_mean, color='r', linestyle='--', label='TRD mean')
plt.fill_between(epsilons, TRD_mean - TRD_std, TRD_mean + TRD_std, alpha=0.1)

plt.xscale('log')
plt.xlabel("Epsilon")
plt.ylabel("Final Loss")
plt.title("Final Loss vs Epsilon (Mean ± Std)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_folder, 'epsilon_vs_loss_mean_std.png'))
plt.close()

# -------- Loss over Steps --------
steps  = data["steps"]
ours_m,  ours_s  = data["ours_m"],  data["ours_s"]
oursF_m, oursF_s = data["oursF_m"], data["oursF_s"]
ATR_m,   ATR_s   = data["ATR_m"],   data["ATR_s"]

plt.figure(figsize=(10, 6))
plt.plot(steps, ours_m, label="Ours")
plt.fill_between(steps, ours_m - ours_s, ours_m + ours_s, alpha=0.2)

plt.plot(steps, oursF_m, label="Ours_F")
plt.fill_between(steps, oursF_m - oursF_s, oursF_m + oursF_s, alpha=0.2)

plt.plot(steps, ATR_m, label="Comparison")
plt.fill_between(steps, ATR_m - ATR_s, ATR_m + ATR_s, alpha=0.2)

plt.axhline(y=TRD_mean, color='r', linestyle='--', label='TRD mean')
plt.fill_between(steps, TRD_mean - TRD_std, TRD_mean + TRD_std, alpha=0.1)

plt.xlabel("Step")
plt.ylabel("Loss (sqrt)")
plt.title("Loss over Steps (Mean ± Std)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_folder, 'loss_over_steps_mean_std.png'))
plt.close()
