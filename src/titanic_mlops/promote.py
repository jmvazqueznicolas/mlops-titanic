"""Promote a registered version to @champion only if it beats the current champion.

    python -m titanic_mlops.promote --version 2
    python -m titanic_mlops.promote --version 2 --force --reason "approved by <name>: ..."

Rule: same data version as the champion, higher accuracy on the SAME test passengers, and
McNemar p < alpha. "Higher by 0.5 points" on ~200 passengers is usually noise.
"""
import argparse

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from titanic_mlops.config import CHAMPION_ALIAS, FEATURES, MODEL_NAME, PARAMS, TARGET
from titanic_mlops.data import load_historical, sha256, split
from titanic_mlops.stats import mcnemar_exact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--force", action="store_true", help="human override; requires --reason")
    ap.add_argument("--reason", default="")
    args = ap.parse_args()
    client = MlflowClient()

    cand = client.get_model_version(MODEL_NAME, args.version)
    cand_hash = client.get_run(cand.run_id).data.params["data_sha256"]
    try:
        champ = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    except MlflowException:
        champ = None

    if champ is None:
        decision, why = True, "no champion yet"
    elif champ.version == cand.version:
        print(f"v{cand.version} is already the champion."); return
    else:
        champ_hash = client.get_run(champ.run_id).data.params["data_sha256"]
        if champ_hash != cand_hash or cand_hash != sha256()[:12]:
            raise SystemExit(f"Not comparable: champion trained on data {champ_hash}, candidate on {cand_hash}, "
                             f"current file {sha256()[:12]}. Retrain both on the same data version.")
        _, test = split(load_historical())
        y = test[TARGET].to_numpy()
        ok = {}
        for tag, mv in (("champion", champ), ("candidate", cand)):
            m = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{mv.version}")
            ok[tag] = m.predict(test[FEATURES]) == y
        acc_c, acc_n = ok["champion"].mean(), ok["candidate"].mean()
        b, c, p = mcnemar_exact(ok["champion"], ok["candidate"])
        print(f"champion  v{champ.version}: accuracy {acc_c:.3f}")
        print(f"candidate v{cand.version}: accuracy {acc_n:.3f}")
        print(f"disagreements: champion right & candidate wrong = {b}, candidate right & champion wrong = {c}; "
              f"McNemar exact p = {p:.3f}")
        decision = acc_n > acc_c and p < PARAMS["promotion"]["alpha"]
        why = (f"beats champion (p = {p:.3f})" if decision else
               f"not distinguishable from champion (p = {p:.3f})" if acc_n > acc_c else "not better than champion")

    if not decision and args.force:
        if not args.reason:
            raise SystemExit("--force needs --reason: an override must say who decided and why.")
        decision, why = True, f"FORCED: {args.reason} (statistical rule said: {why})"

    if decision:
        client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, cand.version)
        client.set_model_version_tag(MODEL_NAME, cand.version, "promotion_reason", why)
        print(f"PROMOTED v{cand.version} to @{CHAMPION_ALIAS}: {why}")
    else:
        client.set_model_version_tag(MODEL_NAME, cand.version, "promotion_reason", f"rejected: {why}")
        print(f"NOT PROMOTED v{cand.version}: {why}. @{CHAMPION_ALIAS} stays on v{champ.version}.")


if __name__ == "__main__":
    main()
