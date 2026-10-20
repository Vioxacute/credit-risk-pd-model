"""
Download the South German Credit dataset and save it as readable CSVs.

Source: South German Credit [Dataset] (2019). UCI Machine Learning Repository.
https://doi.org/10.24432/C5X89F. Licensed CC BY 4.0. It is the corrected
version of the classic Statlog German Credit data (Groemping, 2019).

Writes:
- data/raw/loans.csv: 990 loans with the target column `default`
- data/raw/new_applications.csv: 10 loans held back without the target,
  to demonstrate scoring in predict.py
"""

from __future__ import annotations

import io
import urllib.request
import zipfile

import pandas as pd

from utils import from_root

URL = "https://archive.ics.uci.edu/static/public/522/south+german+credit.zip"

# Original German column names -> English names from the dataset's code table
COLUMNS = {
    "laufkont": "checking_status",
    "laufzeit": "duration_months",
    "moral": "credit_history",
    "verw": "purpose",
    "hoehe": "amount_dm",
    "sparkont": "savings",
    "beszeit": "employment_duration",
    "rate": "installment_rate",
    "famges": "personal_status_sex",
    "buerge": "other_debtors",
    "wohnzeit": "present_residence",
    "verm": "property",
    "alter": "age",
    "weitkred": "other_installment_plans",
    "wohn": "housing",
    "bishkred": "number_credits",
    "beruf": "job",
    "pers": "people_liable",
    "telef": "telephone",
    "gastarb": "foreign_worker",
    "kredit": "credit_risk",
}

# Level codes -> labels, from codetable.txt in the download
LABELS = {
    "checking_status": {1: "no checking account", 2: "< 0 DM", 3: "0-200 DM", 4: ">= 200 DM or salary account"},
    "credit_history": {
        0: "delay in paying off in the past",
        1: "critical account / credits elsewhere",
        2: "no credits taken / all paid back duly",
        3: "existing credits paid back duly",
        4: "all credits at this bank paid back duly",
    },
    "purpose": {
        0: "others", 1: "car (new)", 2: "car (used)", 3: "furniture/equipment", 4: "radio/television",
        5: "domestic appliances", 6: "repairs", 7: "education", 8: "vacation", 9: "retraining", 10: "business",
    },
    "savings": {1: "unknown / none", 2: "< 100 DM", 3: "100-500 DM", 4: "500-1000 DM", 5: ">= 1000 DM"},
    "employment_duration": {1: "unemployed", 2: "< 1 yr", 3: "1-4 yrs", 4: "4-7 yrs", 5: ">= 7 yrs"},
    "installment_rate": {1: ">= 35%", 2: "25-35%", 3: "20-25%", 4: "< 20%"},  # of disposable income
    "personal_status_sex": {
        1: "male: divorced/separated", 2: "female: non-single or male: single",
        3: "male: married/widowed", 4: "female: single",
    },
    "other_debtors": {1: "none", 2: "co-applicant", 3: "guarantor"},
    "present_residence": {1: "< 1 yr", 2: "1-4 yrs", 3: "4-7 yrs", 4: ">= 7 yrs"},
    "property": {1: "unknown / none", 2: "car or other", 3: "savings agreement / life insurance", 4: "real estate"},
    "other_installment_plans": {1: "bank", 2: "stores", 3: "none"},
    "housing": {1: "for free", 2: "rent", 3: "own"},
    "number_credits": {1: "1", 2: "2-3", 3: "4-5", 4: ">= 6"},
    "job": {
        1: "unemployed / unskilled non-resident", 2: "unskilled resident",
        3: "skilled employee / official", 4: "manager / self-employed / highly qualified",
    },
    "people_liable": {1: "3 or more", 2: "0 to 2"},
    "telephone": {1: "no", 2: "yes"},
    "foreign_worker": {1: "yes", 2: "no"},
}


def main() -> None:
    with urllib.request.urlopen(URL) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    df = pd.read_csv(archive.open("SouthGermanCredit.asc"), sep=r"\s+").rename(columns=COLUMNS)

    for col, labels in LABELS.items():
        df[col] = df[col].map(labels)
    df["default"] = 1 - df.pop("credit_risk")  # credit_risk: 1 = good, 0 = bad

    new_apps = df.sample(n=10, random_state=42)
    loans = df.drop(index=new_apps.index)

    out_dir = from_root("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    loans.to_csv(out_dir / "loans.csv", index=False)
    new_apps.drop(columns=["default"]).to_csv(out_dir / "new_applications.csv", index=False)
    print(f"Saved {len(loans)} loans ({loans['default'].mean():.0%} defaulted) and {len(new_apps)} new applications to {out_dir}")


if __name__ == "__main__":
    main()
