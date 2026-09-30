import tkinter as tk
from tkinter import messagebox, ttk

import joblib
import numpy as np
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "yacht_gbr.pkl"
ckpt = joblib.load(MODEL_PATH)
model = ckpt["model"]
feature_names = ckpt["feature_names"]

LABELS = {
    "LongPos":     ("Longitudinal position of CoB", "-5.0 to 0.0"),
    "PrisCoeff":   ("Prismatic coefficient",        "0.53 to 0.60"),
    "LengthDisp":  ("Length-displacement ratio",    "4.34 to 5.14"),
    "BeamDraught": ("Beam-draught ratio",           "2.81 to 5.35"),
    "LengthBeam":  ("Length-beam ratio",            "2.73 to 3.64"),
    "FroudeNum":   ("Froude number",                "0.125 to 0.45"),
}


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Yacht Hydrodynamics — Resistance Predictor")
        self.resizable(False, False)

        header = ttk.Label(self, text="Residuary Resistance Predictor",
                           font=("Helvetica", 13, "bold"))
        header.grid(row=0, column=0, columnspan=3, pady=(14, 8))

        ttk.Separator(self, orient="horizontal").grid(
            row=1, column=0, columnspan=3, sticky="ew", padx=10)

        self.entries = {}
        for r, fname in enumerate(feature_names, start=2):
            label, hint = LABELS[fname]
            ttk.Label(self, text=label, anchor="w", width=30).grid(
                row=r, column=0, padx=(12, 4), pady=5, sticky="w")
            entry = ttk.Entry(self, width=14)
            entry.grid(row=r, column=1, padx=4, pady=5)
            ttk.Label(self, text=hint, foreground="grey", font=("Helvetica", 8)).grid(
                row=r, column=2, padx=(2, 12), sticky="w")
            self.entries[fname] = entry

        ttk.Separator(self, orient="horizontal").grid(
            row=len(feature_names) + 2, column=0, columnspan=3,
            sticky="ew", padx=10, pady=(6, 0))

        ttk.Button(self, text="Predict", command=self.run_predict).grid(
            row=len(feature_names) + 3, column=0, columnspan=3, pady=10)

        self.result_var = tk.StringVar(value="—")
        ttk.Label(self, text="Predicted Resistance:", font=("Helvetica", 10)).grid(
            row=len(feature_names) + 4, column=0, columnspan=1,
            padx=(12, 0), pady=(0, 14), sticky="e")
        ttk.Label(self, textvariable=self.result_var,
                  font=("Helvetica", 13, "bold"), foreground="steelblue").grid(
            row=len(feature_names) + 4, column=1, columnspan=2,
            padx=(4, 12), pady=(0, 14), sticky="w")

    def run_predict(self):
        try:
            values = {f: float(self.entries[f].get()) for f in feature_names}
        except ValueError:
            messagebox.showerror("Input error", "All fields must be numeric.")
            return
        x = np.array([values[f] for f in feature_names]).reshape(1, -1)
        result = float(model.predict(x)[0])
        self.result_var.set(f"{result:.4f}  (per unit weight)")


if __name__ == "__main__":
    App().mainloop()