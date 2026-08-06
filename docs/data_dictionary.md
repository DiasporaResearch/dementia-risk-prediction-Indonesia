# Data Dictionary

Coding scheme for the DEMPRED-ID structured questionnaire, following the same domain structure and Yes/No (1/2) categorical convention used in the team's prior hypertension questionnaires (Nuryunarsih et al., 2023, 2025). Item weights below are for `A1_id`-style unique identification and downstream `01_data_cleaning.py` / `03_aim2_apriori_association.py` processing.

**Predictor domains and priority (per Brain et al., 2024, treemap frequency):** demographic > health/clinical > lifestyle > cognitive > genetic > functioning. Column `priority_tier` below marks whether an item is in the ~20-item consolidated core set (Tier 1) referenced in `PROTOCOL.md` Section 4.4, or the extended ~90-item full set (Tier 2).

## A. Demographic

| Variable | Column name | Type | Coding | Priority tier |
|---|---|---|---|---|
| Age | `age` | Numeric | Years | 1 |
| Sex | `sex` | Categorical | Male=1, Female=2 | 1 |
| Education | `education` | Categorical | Illiterate=1, Primary=2, High school=3, Bachelor+=4 | 1 |
| Occupation | `occupation` | Categorical | Site-specific job categories | 2 |
| Marital status | `marital_status` | Categorical | Single=1, Married=2, Widowed/divorced=3 | 2 |

## B. Clinical / comorbidity

| Variable | Column name | Type | Coding | Priority tier |
|---|---|---|---|---|
| Hypertension | `hypertension` | Categorical | Yes=1, No=2 | 1 |
| Diabetes | `diabetes` | Categorical | Yes=1, No=2 | 1 |
| Cardiovascular disease history | `cvd_history` | Categorical | Yes=1, No=2 | 1 |
| Stroke/TIA history | `stroke_tia` | Categorical | Yes=1, No=2 | 1 |
| BMI | `bmi` | Categorical | <18.5=1, 18.5–24.9=2, 25–29.9=3, ≥30=4 | 1 |
| Hearing impairment | `hearing_impairment` | Categorical | Yes=1, No=2 | 2 |
| Vision impairment | `vision_impairment` | Categorical | Yes=1, No=2 | 2 |
| Depression symptoms | `depression_symptoms` | Categorical | Yes=1, No=2, Sometimes=3 | 1 |
| Systolic BP | `sbp` | Numeric | mmHg | 2 |
| Diastolic BP | `dbp` | Numeric | mmHg | 2 |

## C. Lifestyle

| Variable | Column name | Type | Coding | Priority tier |
|---|---|---|---|---|
| Physical activity | `physical_activity` | Categorical | Yes=1, No=2, Sometimes=3 | 1 |
| Smoking | `smoking` | Categorical | Yes=1, No=2 | 1 |
| Alcohol use | `alcohol_use` | Categorical | Yes=1, No=2 | 2 |
| Sleep quality | `sleep_quality` | Categorical | Poor=1, Fair=2, Good=3 | 2 |
| Sleep duration | `sleep_duration_hrs` | Numeric | Hours/night | 2 |
| Fish consumption | `fish_intake` | Categorical | N=1, Occasionally=2, Sometime=3, Mostly=4, Daily=5 | 1 |
| Vegetable consumption | `vegetable_intake` | Categorical | N=1, Occasionally=2, Sometime=3, Mostly=4, Daily=5 | 1 |
| Fruit consumption | `fruit_intake` | Categorical | N=1, Occasionally=2, Sometime=3, Mostly=4, Daily=5 | 2 |
| Salt intake | `salt_type` | Categorical | Iodized=1, Non-iodized=2 | 2 |
| Fat diet | `fat_diet` | Categorical | Yes=1, No=2 | 2 |

## D. Socioeconomic

| Variable | Column name | Type | Coding | Priority tier |
|---|---|---|---|---|
| Income | `income` | Categorical | Site-specific brackets | 2 |
| Household members | `household_members` | Categorical | ≤5=1, 6–10=2, 11–15=3 | 2 |
| Home ownership | `home_ownership` | Categorical | Yes=1, No=2 | 2 |
| Health insurance access | `health_insurance` | Categorical | Yes=1, No=2 | 2 |

## E. Family / genetic history

| Variable | Column name | Type | Coding | Priority tier |
|---|---|---|---|---|
| Family history of dementia | `family_history_dementia` | Categorical | Yes=1, No=2 | 1 |
| Relative degree (dementia) | `family_history_dementia_degree` | Categorical | 1st-degree=1, 2nd-degree=2 | 2 |
| APOE genotype (optional add-on) | `apoe_status` | Categorical | e4 carrier=1, non-carrier=2, not tested=3 | 2 (optional) |

## F. Symptom / functional status

| Variable | Column name | Type | Coding | Priority tier |
|---|---|---|---|---|
| Memory complaints | `memory_complaints` | Categorical | Yes=1, No=2, Sometimes=3 | 1 |
| IADL functioning | `iadl_score` | Numeric | Standard IADL scale score | 1 |
| ADL functioning | `adl_score` | Numeric | Standard ADL scale score | 2 |

## G. Outcome variables

| Variable | Column name | Type | Coding | Aim |
|---|---|---|---|---|
| MMSE score | `mmse_score` | Numeric | 0–30 | 1 |
| MoCA score | `moca_score` | Numeric | 0–30 | 1 |
| Cognitive impairment status | `cognitive_impairment` | Categorical (derived) | Impaired=1, Not impaired=0 (per validated MMSE/MoCA cutoff for education/age) | 1 |
| IQCODE score | `iqcode_score` | Numeric | Mean item score 1–5 (16-item short form) or 1–5 (26-item long form) | 3 |
| IQCODE decline flag | `iqcode_decline` | Categorical (derived) | Decline=1, No decline=0 (cutoff ≥3.3, confirm against IQCODE version used — see `scripts/04_aim3_iqcode_decline_model.py`) | 3 |

## Notes

- All categorical Yes/No items follow the 1=Yes, 2=No convention from the source hypertension questionnaires, and are recoded to 0/1 in `01_data_cleaning.py::encode_binary()` for modelling.
- Continuous variables must be pre-binned into categorical flags before Aim 2 (apriori) analysis — see `03_aim2_apriori_association.py`.
- This dictionary is a **template**. Finalise exact item wording, response options, and validated Indonesian-language instrument versions (MMSE/MoCA/IQCODE) with the clinical team and ethics committee before fielding.
