"""Aggregate baselines + models; apply the preregistered kill criterion."""
import json
import os

from common import paired_delta_ci

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "outputs")


def load(n):
    p = os.path.join(OUT, f"scores_{n}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    B1, B2 = load("B1_birthyear"), load("B2_rule")
    LOC, COM = load("M_local_qwen14b"), load("M_comm_sonnet5")
    print(f"{'method':22} {'AUC':>7}   95% CI")
    for r in (B1, B2, LOC, COM):
        if r:
            print(f"{r['method']:22} {r['auc']:7.4f}   [{r['ci95'][0]:.4f},{r['ci95'][1]:.4f}]")
    if not COM:
        print("\n(commercial scores not present yet)")
        return
    best_base = max(r["auc"] for r in (B1, B2) if r)
    ci = COM["ci95"]
    gain = COM["auc"] - best_base
    passed = COM["auc"] >= 0.65 and ci[0] > 0.50 and gain >= 0.05
    verdict = "BUILT" if passed else "KILLED-AT-4-not-beaten-or-below-bar"
    print("\n--- PRIMARY (commercial instrument) ---")
    print(f"best deterministic baseline AUC = {best_base:.4f}")
    print(f"M_comm AUC = {COM['auc']:.4f}  CI [{ci[0]:.4f},{ci[1]:.4f}]  gain={gain:+.4f}")
    print(f"need AUC>=0.65, CI>0.5, gain>=+0.05 -> {'PASS' if passed else 'FAIL'}")
    print(f"VERDICT: {verdict}")
    if B2:
        obs, lo, hi, p = paired_delta_ci(COM["labels"], B2["scores"], COM["scores"])
        print(f"paired M_comm - B2 dAUC = {obs:+.4f} [{lo:+.4f},{hi:+.4f}] P(<=0)={p:.3f}")
    if LOC:
        obs, lo, hi, p = paired_delta_ci(COM["labels"], LOC["scores"], COM["scores"])
        print(f"\ncommercial vs local: qwen AUC={LOC['auc']:.4f} sonnet AUC={COM['auc']:.4f}")
        print(f"paired M_comm - M_local dAUC = {obs:+.4f} [{lo:+.4f},{hi:+.4f}] P(<=0)={p:.3f}")
    json.dump({"verdict": verdict, "best_baseline": best_base, "comm_auc": COM["auc"],
               "comm_ci": ci, "gain": gain,
               "local_auc": LOC["auc"] if LOC else None},
              open(os.path.join(OUT, "summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
