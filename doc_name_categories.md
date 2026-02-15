# Document Name Categories Reference

Classification of `doc_name` values from `lookup_table.csv` into granular categories for international investment arbitration documents.

## Summary Statistics

| Metric | Value |
|---|---|
| Total rows | 8,749 |
| Unique doc_names | 5,336 |
| Categories used | 101 |
| "Other" count | 49 (0.6%) |

---

## Categories by Group

### Awards (final tribunal decisions)

| Category | Count | Description | Examples |
|---|---|---|---|
| `award` | 535 | General final/arbitral awards | `arbitral_award`, `award_english`, `final_award` |
| `award_on_jurisdiction` | 73 | Awards specifically on jurisdiction or preliminary objections | `award_on_jurisdiction`, `award_on_jurisdiction_and_admissibility` |
| `partial_award` | 32 | Partial or phased awards | `first_partial_award`, `partial_award`, `third_partial_and_final_award_damages_and_costs` |
| `supplementary_decision` | 27 | Supplementary decisions, addenda, or additional awards | `addendum_to_award`, `supplemental_award_and_interpretation_2004_*` |
| `correction_of_award` | 36 | Corrections, rectifications, or interpretations of awards | `correction_of_the_award`, `rectification_of_award` |
| `interim_award` | 22 | Interim or emergency awards | `interim_award`, `emergency_award` |
| `award_on_costs` | 13 | Awards specifically on costs | `award_on_costs`, `award_in_respect_of_costs` |
| `consent_award` | 8 | Awards embodying settlement agreements | `consent_award`, `award_embodying_the_parties_settlement_agreement_*` |
| `award_on_merits` | 7 | Awards specifically on the merits | `award_on_the_merits`, `award_on_the_merits_of_phase_2` |
| `award_on_damages` | 5 | Awards on damages or quantum | `award_on_damages`, `award_in_respect_of_damages` |
| `resubmission_award` | 2 | Awards in resubmission proceedings | `resubmission_award`, `award_of_the_tribunal_resubmission_proceeding_*` |
| `award_on_liability` | 1 | Awards on jurisdiction and liability | `partial_award_on_liability` |

### Decisions (tribunal/committee rulings)

| Category | Count | Description | Examples |
|---|---|---|---|
| `decision_on_jurisdiction` | 169 | Decisions on jurisdiction, competence, admissibility, or preliminary objections | `decision_on_jurisdiction`, `decision_on_objections_to_jurisdiction` |
| `decision_on_disqualification` | 104 | Decisions on arbitrator disqualification or challenge proposals | `challenge_decision`, `decision_on_the_proposal_to_disqualify_*` |
| `decision_on_annulment` | 95 | Decisions on applications for annulment | `decision_on_annulment`, `decision_on_the_application_for_annulment` |
| `decision_on_stay_of_enforcement` | 68 | Decisions on staying enforcement of awards during annulment | `decision_on_stay_of_enforcement`, `decision_on_continued_stay_of_enforcement` |
| `decision_on_provisional_measures` | 61 | Decisions on provisional or interim measures | `decision_on_provisional_measures`, `decision_on_claimant_s_application_for_interim_measures_*` |
| `decision_other` | 40 | Other tribunal decisions not captured by specific categories | `decision_and_order_by_the_arbitral_tribunal`, `decision_by_tribunal` |
| `decision_on_bifurcation` | 37 | Decisions on bifurcation of proceedings | `decision_on_bifurcation`, `decision_on_bifurcated_jurisdictional_issue` |
| `decision_on_rectification` | 18 | Decisions on correction, interpretation, or rectification requests | `decision_on_rectification`, `decision_on_correction_request` |
| `decision_on_early_dismissal` | 18 | Decisions under Rule 41(5) or equivalent early dismissal provisions | `decision_on_respondent_s_application_under_icsid_arbitration_rule_41_5` |
| `decision_on_liability` | 15 | Decisions on liability or responsibility | `decision_on_liability`, `decision_on_liability_and_heads_of_loss` |
| `decision_on_document_production` | 14 | Decisions on document production requests | `decision_on_document_production`, `decision_on_objections_to_document_production` |
| `decision_on_non_disputing_party` | 14 | Decisions on amicus or non-disputing party applications | `decision_on_authority_to_accept_amicus_submissions`, `decision_on_minbyun_non_disputing_party_application` |
| `decision_on_reconsideration` | 12 | Decisions on reconsideration requests | `decision_on_reconsideration_*`, `decision_on_ecuador_s_reconsideration_motion_*` |
| `decision_on_place_of_arbitration` | 8 | Decisions on seat or place of arbitration | `decision_on_place_of_arbitration`, `decision_on_the_seat_of_arbitration` |
| `decision_on_security_for_costs` | 7 | Decisions on applications for security for costs | `decision_on_security_for_costs`, `decision_on_el_salvador_s_application_for_security_for_costs` |
| `decision_on_costs` | 6 | Decisions on costs or arbitration expenses | `decision_on_costs`, `decision_on_arbitration_expenses_*` |
| `decision_on_revision` | 4 | Decisions on applications for revision | `decision_on_revision_*`, `decision_on_application_for_revision_*` |
| `decision_on_merits` | 3 | Decisions on the merits | `decision_on_merits`, `decision_on_the_merits` |
| `decision_on_consolidation` | 1 | Decisions on consolidation of proceedings | `decision_on_the_constitution_of_the_tribunal` |

### Procedural Orders

| Category | Count | Description | Examples |
|---|---|---|---|
| `procedural_order` | 1,646 | All procedural orders (including annexes, amendments, and tribunal orders) | `procedural_order_no_1`, `amended_procedural_order_*`, `order_no_3` |

### Individual Arbitrator Opinions

| Category | Count | Description | Examples |
|---|---|---|---|
| `dissenting_opinion` | 137 | Dissenting opinions appended to awards or decisions | `dissenting_opinion_of_arbitrator_*`, `statement_of_dissent` |
| `separate_opinion` | 36 | Separate or individual opinions | `separate_opinion`, `individual_opinion_by_committee_member_*` |
| `partial_dissenting_opinion` | 35 | Partially dissenting opinions | `partial_dissenting_opinion_*`, `note_of_partial_dissent_*` |
| `declaration_by_arbitrator` | 19 | Declarations or statements appended by individual arbitrators | `declaration_by_arbitrator_*`, `declaration_of_professor_*` |
| `concurring_opinion` | 17 | Concurring opinions | `concurring_opinion_of_arbitrator_*` |
| `concurring_and_dissenting_opinion` | 12 | Combined concurring and dissenting opinions | `concurring_and_dissenting_opinion`, `concurrent_and_dissent_statement_*` |

### Claimant Submissions

| Category | Count | Description | Examples |
|---|---|---|---|
| `claimant_reply` | 232 | Claimant reply memorials and rebuttals | `claimant_s_reply`, `investor_s_reply_memorial` |
| `claimant_observations` | 214 | Claimant observations, comments, and miscellaneous submissions | `claimant_s_observations_on_*`, `investor_s_submission_on_*` |
| `claimant_request` | 128 | Claimant requests for interim measures, document production, etc. | `claimant_s_request_for_provisional_measures`, `application_for_provisional_measures` |
| `claimant_memorial` | 114 | Claimant memorials on merits, jurisdiction, damages, etc. | `claimant_s_memorial`, `memorial_of_the_investor_*` |
| `claimant_letter` | 95 | Claimant letters to tribunal or ICSID | `claimant_s_letter_to_the_tribunal`, `investor_s_letter_*` |
| `claimant_post_hearing` | 84 | Claimant post-hearing briefs, closing arguments, opening statements | `claimant_s_post_hearing_brief`, `investor_s_closing_memorial` |
| `claimant_rejoinder` | 70 | Claimant rejoinders | `claimant_s_rejoinder`, `canfor_rejoinder_on_jurisdiction` |
| `claimant_press_release` | 67 | Claimant press releases | `claimant_press_release`, `claimant_s_press_release_on_*` |
| `claimant_statement_of_claim` | 65 | Statements of claim (original and amended) | `amended_statement_of_claim`, `claimant_s_statement_of_claim` |
| `claimant_other` | 61 | Other claimant submissions not captured above | `claimant_objections_to_*`, `claimant_response_re_*` |
| `claimant_counter_memorial` | 37 | Claimant counter-memorials (on jurisdiction, annulment) | `claimant_s_counter_memorial_on_jurisdiction` |
| `claimant_submission_on_costs` | 33 | Claimant cost submissions | `claimant_costs_submissions`, `claimant_s_submission_on_costs` |

### Respondent Submissions

| Category | Count | Description | Examples |
|---|---|---|---|
| `respondent_request` | 113 | Respondent requests (security for costs, discovery, etc.) | `respondent_s_request_for_security_for_costs`, `u_s_motion_to_exclude_*` |
| `respondent_observations` | 104 | Respondent observations, comments, and miscellaneous submissions | `respondent_s_observations_on_*`, `argentina_s_response_to_*` |
| `respondent_counter_memorial` | 96 | Respondent counter-memorials | `canada_s_counter_memorial`, `respondent_s_counter_memorial` |
| `respondent_other` | 93 | Other respondent submissions not captured above | `brief_for_the_respondent`, `canada_s_brief_outline_*` |
| `respondent_rejoinder` | 76 | Respondent rejoinders | `canada_s_rejoinder`, `respondent_s_rejoinder` |
| `respondent_letter` | 71 | Respondent letters to tribunal | `canada_s_letter_to_the_tribunal_*`, `u_s_letter` |
| `respondent_post_hearing` | 71 | Respondent post-hearing briefs and submissions | `canada_s_post_hearing_submission`, `u_s_post_hearing_submission` |
| `respondent_reply` | 66 | Respondent replies | `canada_s_reply_memorial_on_jurisdiction`, `u_s_reply_on_jurisdiction` |
| `respondent_preliminary_objection` | 64 | Respondent preliminary objections to jurisdiction | `respondent_s_preliminary_objections`, `u_s_objection_to_jurisdiction` |
| `respondent_press_release` | 54 | Respondent press releases | `colombia_s_press_release_on_final_award_*` |
| `respondent_statement_of_defence` | 46 | Respondent statements of defence/defense | `statement_of_defence`, `canada_s_statement_of_defence_*`, `u_s_statement_of_defense` |
| `respondent_submission_on_costs` | 29 | Respondent cost submissions | `canada_s_submission_on_costs`, `u_s_submission_on_costs` |
| `respondent_request_for_bifurcation` | 28 | Respondent bifurcation requests | `government_of_canada_request_for_bifurcation`, `u_s_request_for_bifurcation` |
| `respondent_memorial_on_jurisdiction` | 25 | Respondent memorials on jurisdiction or preliminary objections | `canada_s_memorial_on_jurisdiction`, `u_s_memorial_on_jurisdiction` |

### Notice Documents

| Category | Count | Description | Examples |
|---|---|---|---|
| `notice_of_arbitration` | 248 | Notices of arbitration and requests for arbitration | `notice_of_arbitration`, `request_for_arbitration` |
| `notice_of_intent` | 106 | Notices of intent to submit claim | `notice_of_intent`, `notice_of_intention_*` |
| `notice_other` | 37 | Other notices (discontinuance, appeal, withdrawal, etc.) | `notice_of_appeal`, `notice_of_discontinuance` |
| `notice_of_dispute` | 24 | Notices of dispute | `notice_of_dispute`, `notice_of_an_investment_dispute` |

### Domestic Court Proceedings

| Category | Count | Description | Examples |
|---|---|---|---|
| `court_judgment` | 405 | Judgments from domestic courts (all jurisdictions) | `amsterdam_district_court_judgment`, `us_court_judgment` |
| `court_decision` | 157 | Decisions from domestic courts | `decision_by_the_dutch_supreme_court`, `swiss_federal_tribunal_decision_*` |
| `court_order` | 137 | Orders from domestic courts | `consent_order_of_the_high_court_*`, `stipulated_order_*` |
| `court_other` | 135 | Other domestic court documents | `appeal_from_the_district_court_*`, `writ_of_summons_*` |
| `court_memorandum_opinion` | 108 | US court memorandum opinions and orders | `memorandum_opinion`, `memorandum_and_order_us_courts` |
| `enforcement_proceeding` | 103 | Petitions/motions to enforce, confirm, or recognize awards | `petition_to_confirm_arbitral_award_*`, `application_for_enforcement` |
| `court_motion` | 68 | Motions filed in domestic court proceedings | `joint_consent_motion_to_stay_*`, `motion_for_default_judgment` |
| `set_aside_proceeding` | 66 | Applications to set aside, vacate, or annul awards in court | `application_to_set_aside_*`, `recours_en_annulation` |
| `court_opinion` | 55 | US court opinions (circuit courts, etc.) | `opinion_and_order_of_the_us_district_court_*` |
| `court_brief` | 7 | Briefs filed in domestic court proceedings | `brief_for_appellant`, `brief_in_opposition` |

### Third-Party / Non-Disputing Party Submissions

| Category | Count | Description | Examples |
|---|---|---|---|
| `non_disputing_party_submission` | 253 | Non-disputing party and Article 1128 submissions | `article_1128_submission`, `submission_of_canada_pursuant_to_article_1128_*` |
| `amicus_curiae` | 126 | Amicus curiae submissions and applications | `amicus_brief_*`, `amicus_curiae_submission_*` |

### Expert Evidence

| Category | Count | Description | Examples |
|---|---|---|---|
| `expert_report` | 198 | Expert reports and valuations | `expert_report_of_*`, `rebuttal_expert_report_*` |
| `witness_statement` | 121 | Witness statements, affidavits, and declarations | `witness_statement_of_*`, `affidavit_of_*`, `witness_declaration_of_*` |

### Hearing Transcripts

| Category | Count | Description | Examples |
|---|---|---|---|
| `hearing_transcript` | 368 | All hearing transcripts (jurisdiction, merits, quantum, etc.) | `hearing_transcript_day_1`, `transcript_day_1`, `oral_hearing_transcript_*` |

### Annulment Proceedings

| Category | Count | Description | Examples |
|---|---|---|---|
| `application_for_annulment` | 30 | Applications for annulment of awards | `application_for_annulment`, `application_for_partial_annulment` |
| `annulment_memorial` | 21 | Memorials on annulment (applicant/respondent) | `applicants_memorial_on_annulment`, `counter_memorial_on_annulment` |

### Administrative / Institutional

| Category | Count | Description | Examples |
|---|---|---|---|
| `tribunal_communication` | 81 | Tribunal letters, emails, and communications to parties | `tribunal_letter`, `tribunal_e_mail_to_parties_*`, `directions_from_the_tribunal_*` |
| `terms_of_appointment` | 45 | Terms of appointment/reference, procedural calendars, rules of procedure | `terms_of_appointment`, `procedural_timetable`, `rules_of_procedure` |
| `pca_communication` | 37 | PCA letters, communications, and press releases | `pca_press_release_*`, `letter_from_the_pca` |
| `confidentiality_order` | 20 | Confidentiality orders and agreements | `confidentiality_order`, `amended_confidentiality_agreement_*` |
| `icsid_communication` | 17 | ICSID letters and communications | `icsid_letter_*`, `registration_notification` |
| `minutes` | 14 | Minutes of sessions or hearings | `minutes_of_the_first_session_*`, `final_minutes_*` |

### Settlement

| Category | Count | Description | Examples |
|---|---|---|---|
| `settlement_agreement` | 11 | Settlement agreements and deeds | `settlement_agreement`, `annex_a_settlement_agreement` |

### Discontinuance

| Category | Count | Description | Examples |
|---|---|---|---|
| `discontinuance_order` | 54 | Orders of discontinuance and termination | `discontinuance_order`, `order_taking_note_of_the_discontinuance_*` |

### Press Releases

| Category | Count | Description | Examples |
|---|---|---|---|
| `press_release` | 53 | Generic press releases (not clearly claimant/respondent) | `press_release`, `news_release_*` |

### Other Document Types

| Category | Count | Description | Examples |
|---|---|---|---|
| `legal_opinion` | 45 | Legal opinions by scholars or independent experts | `legal_opinion_of_*`, `opinion_of_professor_*` |
| `affidavit` | 45 | Affidavits and declarations filed in support of motions | `declaration_of_*`, `affidavit_of_*` |
| `stock_exchange_filing` | 6 | Stock exchange filings | `claimant_s_stock_exchange_filing`, `stock_exchange_filing_*` |
| `errata` | 2 | Errata and corrections to submissions | `errata`, `errata_corrected_petition` |
| `index_of_exhibits` | 2 | Exhibit and authority indices | `cumulative_index_of_supporting_documentation` |
| `annex` | 2 | Standalone annexes | `annex_english`, `annex_a_updated_calendar_*` |

### Miscellaneous

| Category | Count | Description | Examples |
|---|---|---|---|
| `other` | 49 | Uncategorized documents (site visits, contracts, exhibits, newspaper articles, resignations, etc.) | `site_visit_protocol`, `concession_contract_*`, `exhibit_c_2` |

---

## Classification Method

Documents are classified using **ordered rule-based matching** in `classify_doc_names.py`. Rules are applied most-specific-first (first match wins), using substring and regex matching on the `doc_name` field.

Key ordering principles:
1. Hearing transcripts first (before any party submission patterns)
2. Individual opinions (dissenting, concurring, separate) before general award/decision patterns
3. Specific award subtypes before general `award`
4. Specific decision subtypes before general `decision`
5. Claimant/respondent submissions use helper functions to detect party authorship
6. Domestic court documents match on court names and judgment/ruling patterns
7. U.S. as respondent handled separately with NDP (Article 1128) exceptions
8. General fallback to `other` for truly uncategorizable documents
