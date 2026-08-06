# Research Protocol Identifikasi Neurokognitif Gangguan Alzheimer/demensia Terpadu – Indonesia (INGAT-ID)
# Dr Desy Nuryunarsih : 06/08/2026

## Development and Validation of a Multi-Domain Machine Learning Model for Dementia Risk Prediction and Cognitive Decline in Older Adults: A Cross-Sectional Study in Two Hospitals in Indonesia

**Short title:** 
**Principal Investigator:** Desy Nuryunarsih, DDS, MPH, PhD  Faculty of Medical Science, University of St-Andrews, United Kingdom
**Co-Investigators:** [Name, affiliation]; [Name, affiliation]; [Name, affiliation]
**Partner Sites:** [Hospital A, City, Indonesia]; [Hospital B, City, Indonesia]
**Study period:** [Start Month/Year] – [End Month/Year] (12 months)
**Funder:** [Funding body name]

---

## 1. Project Summary

Dementia is a leading cause of disability and dependency in older adults worldwide, yet no validated, context-appropriate risk prediction model currently exists for the Indonesian population. Existing dementia risk models have been developed almost exclusively in high-income, White populations, and external validation studies consistently show poor transportability to low- and middle-income country (LMIC) settings (Stephan et al., 2020; Brain et al., 2024). This project addresses that gap directly.

Building on the team's established methodology for interpretable, tree-based machine learning applied to cardiovascular risk prediction in South Asian populations (Nuryunarsih et al., 2023, 2025, 2024), we propose a single-visit, cross-sectional study of older adults attending two hospitals in Indonesia, structured around three aims from one data-collection wave:

1. **Aim 1**  a machine learning classification model (Decision Tree, Random Forest, XGBoost) predicting current cognitive impairment/dementia status, defined using a validated screening tool (MMSE/MoCA);
2. **Aim 2**  an apriori/market-basket association-rule analysis identifying frequently co-occurring clinical, lifestyle, and socioeconomic attribute combinations associated with cognitive impairment; and
3. **Aim 3**  a retrospective decline-trajectory analysis using the Informant Questionnaire on Cognitive Decline in the Elderly (IQCODE), a validated single-administration informant tool, modelling factors associated with reported cognitive decline over the preceding 10 years.

---

## 2. Background and Significance

### 2.1 The burden of dementia and the need for risk prediction

Dementia is a leading cause of disability and death globally. Identifying individuals at elevated risk is central to prevention, care planning, and health-system resource allocation. Risk-based strategies of this kind have proven effective in cardiovascular disease and stroke prevention, yet no equivalent, widely recommended screening tool exists for dementia (Brain et al., 2024).

### 2.2 The evidence gap in low- and middle-income countries

An updated systematic review co-authored by our team (Brain et al., 2024, *Dementia and Geriatric Cognitive Disorders Extra*) identified 74 dementia risk prediction models published since 2014, incorporating over 450 unique predictor variables. Only five studies developed models specifically in LMIC settings, and external validation of high-income-country models in LMIC populations (e.g., the 10/66 study across seven LMICs) showed generally poor transportability  only the Basic Dementia Risk Model (BDRM), Brief Dementia Screening Indicator (BDSI), and Australian National University Alzheimer's Disease Risk Index (ANU-ADRI) transported adequately (Stephan et al., 2020). No dementia risk model has yet been developed or validated using Indonesian clinical data.

This gap matters because risk factor prevalence, healthcare access, cultural context, and even instrument performance (e.g., olfactory-based tests) differ meaningfully between high-income and LMIC populations (Brain et al., 2024). A model built on Indonesian clinical data, incorporating locally relevant lifestyle, dietary, and socioeconomic predictors, is needed before any risk-stratification tool can be responsibly deployed in Indonesian primary or geriatric care.

### 2.3 Preliminary work and team track record

- **Nuryunarsih et al. (2023,** *Current Hypertension Reports***)**  Naïve Bayes, ANN, logistic regression, and decision tree models predicting SBP/DBP decrease in 100 Indonesian hypertensive patients (≥84% accuracy, ≥90% precision).
- **Nuryunarsih et al. (2025,** *Journal of Current Science and Technology***)**  96-feature, three-algorithm comparison (DT, RF, XGBoost) in 160 hypertensive patients with cardiovascular/diabetic comorbidities in Pakistan; interpretable decision-tree visualisation of medication-response pathways (90% accuracy SBP, 97% DBP).
- **Nuryunarsih et al. (2024,** *Journal of Medical Artificial Intelligence***)**  apriori/market-basket association rule mining identifying co-occurring lifestyle, symptom, and health-status attributes in 98 hypertension/HTN-CVD patients in Pakistan.

This project transfers that validated methodology  structured multi-domain questionnaire design, tree-based interpretable ML, and association-rule mining  from cardiovascular risk to dementia risk.

---

## 3. Research Aims and Objectives

**Overall aim:** To develop and internally validate an interpretable, multi-domain machine learning framework for dementia risk and cognitive decline profiling in older adults attending two hospitals in Indonesia, using a single cross-sectional data-collection wave.

### Aim 1  Current cognitive status prediction
Develop and compare Decision Tree, Random Forest, and XGBoost classifiers predicting current cognitive impairment status (MMSE/MoCA cutoffs) from demographic, clinical, lifestyle, socioeconomic, and family-history predictors; evaluate via accuracy, sensitivity, specificity, F1-score, AUC, and calibration; generate interpretable decision-tree visualisations.

### Aim 2  Risk-factor association profiling
Apply apriori/market-basket analysis (minimum support ≥60%, as in Nuryunarsih et al., 2024) to identify frequently co-occurring attribute combinations associated with cognitive impairment.

### Aim 3  Retrospective decline-trajectory profiling
Using IQCODE informant ratings collected at the same visit, identify predictors most strongly associated with informant-reported cognitive decline over the preceding decade.

### Future phase (not funded under this proposal)
A planned Phase 2 prospective cohort extension will re-contact consenting participants at 2–3 years to enable true incident-dementia modelling and external validation of the Phase 1 models.

---

## 4. Methods

### 4.1 Study design
Cross-sectional, single-visit, two-site clinical study, following the design precedent of Nuryunarsih et al. (2023, 2025, 2024).

### 4.2 Setting
[Hospital A, City] and [Hospital B, City], Indonesia. [Add justification: geriatric/neurology outpatient clinics, catchment population, prior collaboration history.]

### 4.3 Participants

**Inclusion criteria:** adults aged ≥60 years attending general medicine, geriatric, or neurology outpatient services at either site, with an available informant (spouse, adult child, or co-resident caregiver) able to complete the IQCODE, and who provide written informed consent (Bahasa Indonesia).

**Exclusion criteria:** pre-existing formal diagnosis of dementia or major neurocognitive disorder at enrolment; acute delirium; severe uncorrected sensory impairment precluding cognitive testing; other severe psychiatric or neurological illness confounding assessment.

### 4.4 Sample size and statistical power

Sample size was calculated using the three-criterion approach of Riley et al. (2020) for developing a binary-outcome clinical prediction model  the same method cited in Nuryunarsih et al. (2025). The three criteria are:

- **(i)** global shrinkage factor S ≥ 0.90 (limiting overfitting of predictor effects to ≤10%);
- **(ii)** an absolute difference of ≤0.05 between apparent and optimism-adjusted Nagelkerke's R²;
- **(iii)** precise estimation of the overall outcome risk (prevalence), margin of error ≤0.05.

The anticipated Cox-Snell R² required for criteria (i) and (ii) was derived via Monte Carlo calibration to a target C-statistic (see `scripts/00_sample_size_calculation.py` and `docs/sample_size_calculation.md` for full method and reproducible code).

**Assumptions:** anticipated cognitive impairment prevalence φ = 0.20; target C-statistic = 0.80 ("good" discrimination per Brain et al., 2024's own classification: <0.70 poor, 0.70–<0.80 fair, 0.80–<0.90 good, ≥0.90 excellent); target shrinkage S = 0.90; target precision δ = 0.05.

| Candidate predictors (p) | Criterion (i): shrinkage | Criterion (ii): overfitting | Criterion (iii): precision | **Required n** | EPP |
|---|---|---|---|---|---|
| 20 | 891 | 72 | 246 | **892** | 8.9 |
| 25 | 1,108 | 91 | 246 | **1,109** | 8.9 |
| 30 (base case) | 1,330 | 109 | 246 | **1,330** | 8.9 |
| 40 | 1,763 | 146 | 246 | **1,763** | 8.8 |
| 50 | 2,211 | 182 | 246 | **2,211** | 8.8 |

**Worked example (simplified, base case):** the shrinkage criterion collapses to `n ≈ p / [(S−1) × ln(1 − R²/S)]`. With S=0.90 and R²≈0.181 (from the C-stat=0.80 simulation), the denominator evaluates to ≈0.02246, so `n ≈ p / 0.02246 ≈ p × 44.5`  i.e., **each additional predictor costs ≈44–45 participants** at these assumptions. This multiplier grows sharply for weaker anticipated models (≈109/predictor at C=0.70) and shrinks for stronger ones (≈22/predictor at C=0.90).

Across all scenarios, criterion (i) is the binding constraint, not the precision-of-prevalence criterion  a materially larger requirement than the traditional EPP≥10 heuristic, and much larger than the EPP≈1.3–1.4 accepted (with caveats) in the team's earlier hypertension study. Two complementary strategies are proposed:

1. **Predictor consolidation:** reduce the candidate predictor set from the full ~90-item questionnaire to a literature- and expert-panel-prioritised core set of ≈20 predictors (drawing on the highest-frequency variables identified in Brain et al., 2024, Fig. 2), lowering the target to n ≈ 900.
2. **Phased/adaptive recruitment:** enrol continuously across the 12-month window at both sites with a target of n = 900–1,000 (split per expected site volume  see Section 4.4.1); if full recruitment is not reached, the achieved sample will be explicitly reported against the Riley criteria actually met, and treated as a pilot/feasibility dataset feeding into the Phase 2 proposal.

Final calculation will be re-run using `pmsampsize` (R or Python) once pilot prevalence data are available.

#### 4.4.1 Per-site allocation

At the consolidated-predictor target (n ≈ 900), an even 50/50 split across two sites gives **≈450 participants per hospital**; at the base-case (30-predictor) target (n ≈ 1,330), an even split gives **≈665 per hospital**. An even split assumes comparable eligible-patient volume at both sites  confirm relative catchment size before finalising the per-site target, and consider inflating the total by 10–15% to buffer against ineligibility, refusal, or missing-informant dropout (e.g., 900 × 1.15 ≈ 1,035 total, ≈518/site).

### 4.5 Outcome measures

| Aim | Outcome | Instrument |
|---|---|---|
| 1 | Current cognitive impairment status (binary/graded) | MMSE and/or MoCA (Indonesian-validated version) |
| 2 | Attribute co-occurrence patterns | Derived from full questionnaire item set |
| 3 | Retrospective 10-year decline trajectory | IQCODE (informant-rated, short or long form) |

### 4.6 Predictor variables

Informed by the predictor treemap in Brain et al. (2024)  age, sex, education, diabetes, BMI, hypertension, smoking, APOE status, depression, hyperlipidaemia, and physical activity as the most frequently used predictors across 74 published models  and the team's prior 96-feature HTN questionnaire, the structured questionnaire captures:

- **Demographic:** age, sex, education level, occupation, marital status
- **Clinical/comorbidity:** hypertension, diabetes, cardiovascular disease, stroke/TIA history, BMI, hearing/vision impairment, depression symptoms
- **Lifestyle:** physical activity, smoking, alcohol use, sleep quality/duration, diet (fish, vegetable, fruit, salt, fat intake)
- **Socioeconomic:** income, household composition, home ownership, health insurance access
- **Family/genetic history:** parental/sibling history of dementia or cognitive impairment
- **Symptom/functional status:** memory complaints, IADL/ADL functioning, mood symptoms

APOE genotyping is listed as an optional add-on given cost constraints but strong predictive value in the literature.

See `questionnaire/questionnaire_template.md` and `docs/data_dictionary.md` for full item-level detail.

### 4.7 Data collection procedure

Trained research assistants will administer the structured questionnaire and cognitive screening tools face-to-face in Bahasa Indonesia, following written informed consent. The informant IQCODE will be completed concurrently by an accompanying family member/caregiver. Data will be recorded into a structured electronic case report form (REDCap or equivalent).

### 4.8 Statistical and machine learning analysis

- Data cleaning: missing data via median/clinically-informed imputation; outliers via IQR-based bounds.
- Descriptive statistics and correlation analysis (Pearson/point-biserial) between candidate predictors and outcomes.
- **Aim 1:** Decision Tree (CART, Gini index), Random Forest, XGBoost; 80/20 train-test split; 5-fold cross-validation; grid-search hyperparameter tuning; class-imbalance weighting; accuracy, sensitivity, specificity, precision, F1-score, AUC; decision-tree visualisation via Graphviz/Pydotplus.
- **Aim 2:** Apriori algorithm (mlxtend/apyori, Python), minimum support ≥60%, reporting antecedent/consequent support and confidence.
- **Aim 3:** Multivariable regression and/or tree-based modelling of IQCODE decline scores against the predictor set.
- **Calibration:** calibration plots, calibration slope/intercept, Brier score reported alongside discrimination  addressing a gap found in only 20/74 models reviewed in Brain et al. (2024).
- **Benchmark comparison:** BDSI, BDRM, and ANU-ADRI scores calculated directly in the sample using existing questionnaire items, as the three models shown to transport adequately into LMIC (10/66) settings (Brain et al., 2024)  the most defensible existing benchmarks.
- All analyses in Python 3.x (pandas, scikit-learn, XGBoost, mlxtend).

### 4.9 Ethical considerations

Ethical approval will be sought from [Institutional Ethics Committee name(s)]. Written informed consent (Bahasa Indonesia) will be obtained from all participants and informants, following the Declaration of Helsinki (2013), consistent with the team's prior approvals (e.g., PMAS-AAUR/1406). Participants screening positive for likely dementia will be referred for full diagnostic work-up as a condition of participation  this study is a research risk-model development exercise, not a diagnostic service.

### 4.10 Reporting standards

Brain et al. (2024) found only 10/74 reviewed studies (13.5%) followed TRIPOD or STROBE guidelines. This study will be conducted and reported in full accordance with **TRIPOD** (Collins et al., 2015) for model-development components (Aims 1 and 3), and **STROBE** (von Elm et al., 2008) for the cross-sectional/association-rule components (Aim 2). The protocol will be pre-registered (PROSPERO or OSF) prior to data collection, and a completed TRIPOD checklist (`docs/TRIPOD_checklist.md`) will accompany the resulting manuscript(s).

---

## 5. Innovation

- First multi-domain, interpretable ML dementia risk profiling study in an Indonesian clinical population, addressing the LMIC evidence gap identified in Brain et al. (2024).
- Combines three complementary analytic lenses (classification, association-rule mining, decline-trajectory modelling) from a single data-collection wave.
- Extends a track record of interpretable, tree-based (not "black-box") modelling producing clinically actionable decision pathways.
- Incorporates locally relevant lifestyle/dietary predictors shown in the team's own prior research to carry independent predictive value.
- Directly benchmarks against BDSI, BDRM, and ANU-ADRI rather than developing in isolation.
- Prospective TRIPOD/STROBE compliance and calibration reporting, addressing two specific quality gaps identified in Brain et al. (2024).

---

## 6. Expected Outcomes and Significance

This project will produce: (i) the first Indonesia-derived, multi-domain dementia risk classification model with reported discrimination and calibration; (ii) clinically interpretable risk-factor association rules for patient education and decision support; (iii) a decline-trajectory risk profile usable as a low-cost outpatient screening adjunct; and (iv) a validated data-collection pipeline seeding a Phase 2 prospective study. Findings will inform locally relevant, resource-appropriate dementia risk-reduction strategies, consistent with the Lancet Commission on Dementia Prevention (Livingston et al., 2020).

---

## 7. Timeline

| Milestone | Timing |
|---|---|
| Ethics approval submission and approval | Months 1–3 |
| Research assistant training; instrument piloting (n≈20) | Months 3–4 |
| Data collection at both hospital sites | Months 4–8 |
| Data cleaning, coding, correlation analysis | Months 8–9 |
| ML model development and validation (Aims 1–3) | Months 9–11 |
| Manuscript preparation and Phase 2 grant development | Months 11–12 |

---

## 8. Investigator Team and Roles

- **Desy Nuryunarsih (PI)**  conceptualisation, methodology, ML analysis, manuscript preparation, overall leadership.
- **[Co-investigator names]**  site coordination, data curation, ethics liaison, participant recruitment, project administration.
- **[Clinical collaborator]**  clinical oversight, diagnostic referral pathway for screen-positive participants.

---

## 9. Limitations

- Cross-sectional design cannot establish true incident dementia risk or causal direction; the IQCODE-based decline measure is a validated but retrospective, informant-dependent proxy.
- Single-country, two-site sampling may limit generalisability; external validation, ideally in additional Indonesian regions or LMIC settings, will be required.
- Questionnaire/screening-tool cognitive classification may introduce outcome misclassification relative to full neuropsychological/clinical diagnosis; referral pathways partially mitigate but do not eliminate this.
- The predictor-to-event ratio will be monitored against Riley et al. (2020) criteria and may require further feature consolidation if the achieved sample is smaller than targeted.

---

## References

- Anstey, K. J., Cherbuin, N., Herath, P. M., et al. (2014). A self-report risk index to predict occurrence of dementia in three independent cohorts of older adults: the ANU-ADRI. *PLoS ONE*, 9(1), e86141.
- Barnes, D. E., Beiser, A. S., Lee, A., et al. (2014). Development and validation of a brief dementia screening indicator for primary care. *Alzheimer's & Dementia*, 10(6), 656–665.
- Brain, J., Kafadar, A. H., Errington, L., et al. (2024). What's new in dementia risk prediction modelling? An updated systematic review. *Dementia and Geriatric Cognitive Disorders Extra*, 14, 49–74. https://doi.org/10.1159/000539744
- Collins, G. S., Reitsma, J. B., Altman, D. G., & Moons, K. G. M. (2015). Transparent reporting of a multivariable prediction model for individual prognosis or diagnosis (TRIPOD). *BMC Medicine*, 13, 1.
- Licher, S., Leening, M. J., Yilmaz, P., et al. (2019). Development and validation of a dementia risk prediction model in the general population. *American Journal of Psychiatry*, 176(7), 543–551.
- Livingston, G., Huntley, J., Sommerlad, A., et al. (2020). Dementia prevention, intervention, and care: 2020 report of the Lancet Commission. *The Lancet*, 396(10248), 413–446.
- Nuryunarsih, D., Herawati, L., Badi'ah, A., Donsu, J. D. T., & Okatiranti. (2023). Predicting changes in systolic and diastolic blood pressure of hypertensive patients in Indonesia using machine learning. *Current Hypertension Reports*, 25(11), 377–383.
- Nuryunarsih, D., Wahyuningsih, H. P., Rauf, S., Zaidan, M., & Herawati, L. (2024). Utilizing an apriori algorithm to examine attributes associated with hypertension and hypertension cardiovascular patients in Pakistan. *Journal of Medical Artificial Intelligence*, 7, 34.
- Nuryunarsih, D., Rauf, S., Okatiranti, et al. (2025). Predicting systolic and diastolic blood pressure response using machine learning: A 96-feature analysis in hypertensive patients with comorbidities. *Journal of Current Science and Technology*, 15(4), Article 140.
- Riley, R. D., Ensor, J., Snell, K. I. E., et al. (2020). Calculating the sample size required for developing a clinical prediction model. *BMJ*, 368, m441.
- Stephan, B. C. M., Pakpahan, E., Siervo, M., et al. (2020). Prediction of dementia risk in low-income and middle-income countries (the 10/66 Study). *The Lancet Global Health*, 8(4), e524–e535.
- von Elm, E., Altman, D. G., Egger, M., et al. (2008). The STROBE statement: guidelines for reporting observational studies. *Journal of Clinical Epidemiology*, 61(4), 344–349.
