"""APOE source audit and explicit decoding.

Root cause of the v1 'APOE_genotype_decoded_N = 0' problem: APOE_reported holds bare numeric codes
(e.g. 0/1/2). Those are never guessed; they are decoded only through an explicit map whose encoding is
declared (epsilon4_dosage or epsilon4_carrier). Cohort profiles in profiles/*.json supply that map so the
end-to-end run decodes APOE automatically, and the audit table shows exactly what was decoded.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .common import read_json, read_tsv, sha256, validate_ids, write_json, write_tsv


def normalized_token(value):
    value = str(value).strip()
    if re.fullmatch(r"-?\d+\.0+", value):
        value = value.split(".")[0]
    return value


def decode_genotype(value):
    """Recognisable genotype strings only (e3/e4, 3/4, 44 ...). Single digits are NOT genotypes."""
    token = normalized_token(value).upper().replace("Ε", "E").replace("Ɛ", "E")
    token = token.replace("APOE", "").replace("E", "").replace("*", "").replace(" ", "")
    match = re.fullmatch(r"([234])(?:[/|,;:_-]?)([234])", token)
    return float(sum(x == "4" for x in match.groups())) if match else np.nan


def prepare_apoe(pheno, cfg, output, map_path=None, source_overrides=None):
    pheno = pheno.copy()
    if "APOE4" not in pheno:
        pheno["APOE4"] = np.nan
    pheno["APOE4"] = pd.to_numeric(pheno["APOE4"], errors="coerce")
    valid = pheno.APOE4.dropna()
    if not valid.isin([0, 1, 2]).all():
        raise ValueError("Prepared APOE4 values must be 0,1,2 or missing")
    overrides = {k: v for k, v in (source_overrides or {}).items() if v is not None}
    if set(overrides) - {"source", "column", "fid_column", "iid_column"}:
        raise ValueError("Unsupported APOE source override")
    if overrides and not map_path:
        raise ValueError("APOE source overrides require --apoe-map with documented value encoding")
    saved_headers = cfg.get("columns", {})
    source = Path(overrides.get("source", cfg.get("phenotype_file", ""))).expanduser().resolve()
    headers = {"fid": overrides.get("fid_column", saved_headers.get("fid", "")),
               "iid": overrides.get("iid_column", saved_headers.get("iid", "")),
               "apoe_genotype": overrides.get("column", saved_headers.get("apoe_genotype", ""))}
    audit = {"prepared_decoded_N": int(pheno.APOE4.notna().sum()),
             "saved_source_file": cfg.get("phenotype_file", ""),
             "saved_source_headers": {k: saved_headers.get(k, "") for k in headers},
             "source_file": str(source), "source_headers": headers,
             "source_column": headers["apoe_genotype"], "source_overrides": overrides,
             "mapping_file": str(Path(map_path).resolve()) if map_path else None,
             "warnings": [], "status": "checking"}

    def fail(message):
        audit.update(status="failed", error=message)
        write_json(output / "apoe_audit.json", audit)
        raise ValueError(message)

    write_json(output / "apoe_audit.json", audit)
    print(f"APOE source: {source}; FID={headers['fid']!r}; IID={headers['iid']!r}; APOE={headers['apoe_genotype']!r}", flush=True)
    raw_tokens = None
    counts = pd.DataFrame(columns=["SOURCE_VALUE", "N_COHORT", "GENOTYPE_E4_DOSAGE_IF_RECOGNIZED"])
    column = audit["source_column"]
    if source.is_file() and column:
        raw = read_tsv(source)
        needed = [headers["fid"], headers["iid"], column]
        audit["available_source_headers"] = raw.columns.tolist()
        if not set(needed).issubset(raw):
            audit["missing_source_headers"] = [x for x in needed if x not in raw]
            audit["source_error"] = (f"Missing configured headers {audit['missing_source_headers']!r} in {source}. "
                                     f"Available headers: {raw.columns.tolist()!r}")
        else:
            if len(set(needed)) != 3:
                fail("APOE source FID, IID and APOE columns must be distinct")
            raw = raw[needed].copy()
            raw.columns = ["FID", "IID", "APOE_SOURCE"]
            for key in ["FID", "IID"]:
                raw[key] = raw[key].str.strip()
            validate_ids(raw, "APOE phenotype source")
            pairs = pheno[["FID", "IID"]].merge(raw, on="IID", how="left", suffixes=("", "_raw"), validate="one_to_one")
            wrong = pairs.FID_raw.notna() & pairs.FID.ne(pairs.FID_raw)
            if wrong.any():
                fail("APOE source FID/IID pairs disagree with prepared phenotypes")
            audit["source_matched_N"] = int(pairs.FID_raw.notna().sum())
            audit["source_unmatched_N"] = int(pairs.FID_raw.isna().sum())
            if map_path and not audit["source_matched_N"]:
                fail("No APOE source IDs match the prepared cohort; check --apoe-fid-column and --apoe-iid-column")
            raw_tokens = pairs.APOE_SOURCE.fillna("").map(normalized_token)
            counts = raw_tokens.value_counts(dropna=False).rename_axis("SOURCE_VALUE").reset_index(name="N_COHORT")
            counts["GENOTYPE_E4_DOSAGE_IF_RECOGNIZED"] = counts.SOURCE_VALUE.map(decode_genotype)
            audit["source_sha256"] = sha256(source)
    else:
        audit["source_error"] = (f"APOE phenotype file does not exist: {source}" if not source.is_file()
                                 else "Saved configuration has no APOE source column")
    missing = {normalized_token(x) for x in cfg.get("missing_values", ["", "NA", "-9", "."])}
    template = {"encoding": "SET_TO_epsilon4_dosage_OR_epsilon4_carrier",
                "mapping": {str(t): None for t in counts.SOURCE_VALUE if t not in missing and not np.isfinite(decode_genotype(t))}}
    write_json(output / "apoe_mapping_template.json", template)
    write_json(output / "apoe_audit.json", audit)
    mapping, kind = {}, None
    if map_path:
        if raw_tokens is None:
            fail("Cannot apply an APOE map: " + audit["source_error"] +
                 ". The completed run's saved configuration supplies defaults. Use --apoe-column, "
                 "--apoe-source, --apoe-fid-column and/or --apoe-iid-column (or the cohort profile) "
                 "to override only APOE input; see apoe_audit.json.")
        spec = read_json(map_path)
        if set(spec) != {"encoding", "mapping"} or not isinstance(spec["mapping"], dict):
            fail("APOE map needs exactly encoding and mapping fields")
        kind = spec["encoding"]
        if kind not in {"epsilon4_dosage", "epsilon4_carrier"}:
            fail("Set APOE map encoding to epsilon4_dosage or epsilon4_carrier from the source data dictionary. "
                 "Encoding describes the values, not the column name; select the column using --apoe-column.")
        allowed = {0, 1, 2} if kind == "epsilon4_dosage" else {0, 1}
        for token, value in spec["mapping"].items():
            token = normalized_token(token)
            if token in mapping:
                fail("Duplicate APOE keys after normalization")
            if value is not None and (type(value) is not int or value not in allowed):
                fail(f"Invalid mapped APOE value for {token!r}")
            mapping[token] = np.nan if value is None else value
        unknown = set(raw_tokens) - missing - set(mapping)
        unknown = sorted(x for x in unknown if not np.isfinite(decode_genotype(x)))
        if unknown:
            fail(f"Unmapped APOE source values: {unknown}; no coding is guessed")

        def decode(token):
            if token in missing:
                return np.nan
            if token in mapping:
                return mapping[token]
            value = decode_genotype(token)
            return value if kind == "epsilon4_dosage" else float(value > 0)
        values = raw_tokens.map(decode).to_numpy(float)
        pheno["APOE4"] = values if kind == "epsilon4_dosage" else np.nan
        pheno["APOE_CARRIER"] = np.where(np.isfinite(values), (values > 0).astype(float), np.nan)
        audit.update(mapping_encoding=kind, mapping_sha256=sha256(map_path))
    else:
        pheno["APOE_CARRIER"] = pheno.APOE4.gt(0).astype(float).where(pheno.APOE4.notna())
    if len(counts):
        def decoded(token):
            if token in missing:
                return "missing"
            if token in mapping:
                return "unavailable" if not np.isfinite(mapping[token]) else str(int(mapping[token]))
            g = decode_genotype(token)
            return "unmapped" if not np.isfinite(g) else str(int(g))
        counts["DECODED_AS"] = counts.SOURCE_VALUE.map(decoded)
        counts["DECODED_MEANING"] = ("epsilon4 dosage" if kind == "epsilon4_dosage" else
                                     "epsilon4 carrier" if kind == "epsilon4_carrier" else "no map supplied")
    write_tsv(output / "apoe_source_value_counts.tsv", counts)
    for dosage in (1, 2):
        pheno[f"APOE_{dosage}"] = pheno.APOE4.eq(dosage).astype(float).where(pheno.APOE4.notna())
    dose = pheno.APOE4.dropna()
    if len(dose):
        freq = float(dose.sum() / (2 * len(dose)))
        audit["epsilon4_allele_frequency"] = freq
        audit["dosage_counts"] = {str(int(k)): int(v) for k, v in dose.value_counts().sort_index().items()}
        if not 0.03 <= freq <= 0.35:
            audit["warnings"].append(f"Implausible epsilon4 allele frequency {freq:.3f}; check the APOE map against the data dictionary.")
        known = pheno.AD.notna() & pheno.APOE4.notna() if "AD" in pheno else None
        if known is not None and known.any():
            by = pheno[known].groupby("AD").APOE4.mean()
            if 0 in by and 1 in by:
                audit["mean_e4_dosage_controls"], audit["mean_e4_dosage_cases"] = float(by[0]), float(by[1])
                if by[1] < by[0]:
                    audit["warnings"].append("Cases carry fewer epsilon4 alleles than controls; the map may be reversed.")
    for message in audit["warnings"]:
        print("WARNING: " + message, flush=True)
    audit.update(dosage_available_N=int(pheno.APOE4.notna().sum()), carrier_available_N=int(pheno.APOE_CARRIER.notna().sum()))
    audit["status"] = "available" if audit["carrier_available_N"] else "unavailable_APOE_models_will_be_reported_as_skipped"
    write_json(output / "apoe_audit.json", audit)
    return pheno, audit
