# Disease codes: keeping the prediction target small

Disease codes are the outcomes the model predicts; labs and vitals are context. The model
predicts the next outcome over the whole target vocabulary, so the size and granularity of that
vocabulary matter. Delphi predicts about 1,300 tokens (ICD-10 three-character codes from
chapters I to XVII, plus death). A source coded at full detail can give tens of thousands.

## How the coding systems are organized

| System | Structure | Size | Where we meet it |
|---|---|---|---|
| ICD-10 (WHO) | A strict tree: 22 chapters → blocks → three-character categories (E11, type 2 diabetes) → four-character subcategories (E11.9) | A few thousand categories; Delphi uses 1,256 of them | UK Biobank, Delphi |
| ICD-10-CM (US) | ICD-10 extended to up to seven characters | About 74,000 codes (FY2025) | MIMIC-IV from October 2015 |
| ICD-9-CM | Three-digit categories plus up to two decimals | About 14,000 codes | MIMIC-IV before October 2015 |
| SNOMED CT | A polyhierarchy: concepts linked by "is a" relationships, a concept can have several parents, 19 top-level hierarchies | 360,000+ concepts | Synthea |
| Phecodes | Groups of ICD codes built for EHR phenotype research, with public maps from ICD-9-CM, ICD-10-CM and ICD-10 | About 1,850 groups (v1.2) | Optional grouping |

SNOMED CT is hierarchical, but not as a single tree with fixed levels, so "cut at the third
level" is not defined the way it is for ICD-10. Official maps from SNOMED CT to ICD-10 exist
(SNOMED International, and NLM for ICD-10-CM) but need a license and cannot be redistributed.

## What our data looks like

| Dataset | Disease codes | After grouping | Targets after the frequency filter |
|---|---|---|---|
| Synthea (22,962 patients) | 919 SNOMED condition codes, 210 of them disorders | none needed | 163 diseases + death |
| MIMIC-IV demo (100 patients) | 1,474 ICD codes (734 ICD-10, 740 ICD-9) | 704 three-character categories (379 ICD-10, 325 ICD-9) | 142 + death (at least 5 of 80 training patients) |
| MIMIC-IV diagnoses (364,627 patients) | 28,583 ICD codes | `icd_gems`: 1,875 categories. `icd_3char`: 2,842 (ICD-9 and ICD-10 apart). `phecode`: 2,267 | `icd_gems`: 1,017 + death (970 ICD-10, 47 ICD-9 the GEMs cannot map). `icd_3char`: 1,397 + death. `phecode`: 1,360 + death (1,214 phecodes). At least 100 of 262,187 training patients |

Synthea's disease set is small because its disease modules are, so the problem Jun raised does
not show up there. It shows up on real data: even 100 MIMIC patients carry 1,474 distinct codes.

## What the framework does

1. **Inputs stay fine-grained, targets are grouped.** Two vocabularies; the grouping is a registered
   function chosen in the data config, and `modlehr prepare` prints, per rule, how many codes were
   matched, how many targets they became and how many passed the frequency filter.
2. **ICD-coded sources (MIMIC-IV, eICU): ICD-10 three-character categories**, the level Delphi uses.
   `icd_3char` truncates ICD-9 and ICD-10 separately, so the two eras of MIMIC-IV form two target
   families. `icd_gems` maps each ICD-9 code through the CMS General Equivalence Mappings (public)
   before truncating, so both eras land in one set; an ICD-9 code with several ICD-10 targets takes
   the category most of them share, and codes the GEMs cannot map stay ICD-9 categories. The MIMIC-IV
   configs use `icd_gems`; `configs/data/mimic_iv_core_icd3char.yaml` keeps the eras apart for
   comparison.
3. **Phecodes**: `phecode` maps ICD-9-CM and ICD-10-CM through the PheWAS catalog maps (public) into
   about 1,850 clinically meaningful groups, looking up ever shorter prefixes for codes more specific
   than the map; unmapped codes are kept as ICD categories or dropped. To compare against ICD-10
   categories on MIMIC-IV.
4. **SNOMED-coded sources: map to ICD-10** where a licensed map is available; otherwise group by
   "is a" ancestors to a curated set. Synthea does not need either.
5. **The frequency filter** (`min_subjects_per_token`) drops outcomes too rare to learn and caps the
   target vocabulary on every source.

The mapping files are public but not in the repository. `scripts/fetch_code_maps.sh` downloads
them into `data/maps/` and checks each against the hash of the copy the configs were made with:

| File | Source | Contents |
|---|---|---|
| `data/maps/gems/2018_I9gem.txt` | CMS, 2018 ICD-10-CM General Equivalence Mappings | 24,860 ICD-9-CM → ICD-10-CM rows; 14,145 ICD-9 codes with a map |
| `data/maps/phecode/Phecode_map_v1_2_icd9_icd10cm.csv` | PheWAS catalog, Phecode Map 1.2 (Combined) | 14,516 ICD-9-CM and 82,924 ICD-10-CM codes; a `Flag` column tells the versions apart |

`configs/data/mimic_iv_core_gems.yaml` and `configs/data/mimic_iv_core_phecode.yaml` use them:

```yaml
target_grouping: icd_gems
target_grouping_options: {gems_path: data/maps/gems/2018_I9gem.txt}
```

```yaml
target_grouping: phecode
target_grouping_options:
  icd9_map: data/maps/phecode/Phecode_map_v1_2_icd9_icd10cm.csv   # the combined file serves both
  icd10_map: data/maps/phecode/Phecode_map_v1_2_icd9_icd10cm.csv
  unmapped: keep
```

Two things the combined phecode map needs care with. Eleven codes exist in both ICD versions with
different meanings (ICD-9 E858 is a drug poisoning, ICD-10-CM E85.8 an amyloidosis), so each version
reads only its own rows. And 7,821 ICD-10-CM codes are listed with more than one phecode (carbon
monoxide poisoning, T58, under both a toxic-effect and a poisoning phecode); the loader takes the
first one listed, since a target grouping gives each code one outcome.
