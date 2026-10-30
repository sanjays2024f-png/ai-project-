"""Replot learning curve with HONEST numbers (train = 3-fold CV, not train-on-train)."""
from pathlib import Path
RES = Path(__file__).resolve().parents[2] / "results" / "figures"
TR = [0.4678, 0.4862, 0.5863, 0.5975, 0.6231]
VA = [0.251, 0.1472, 0.3225, 0.6193, 0.6228]
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
xs = [20, 40, 60, 80, 100]
plt.figure()
plt.plot(xs, TR, marker="o", label="train F1 (honest 3-fold CV)")
plt.plot(xs, VA, marker="s", label="val F1")
plt.xlabel("train data %")
plt.ylabel("macro F1")
plt.title("Learning curve (good): honest-train=0.623 val=0.623 gap=0.000")
plt.legend()
lp = RES / "learning_curve_measured.png"
plt.savefig(lp, dpi=100)
plt.close()
print("PLOT_OK", lp)
