"""
Classify doc_name values in lookup_table.csv into granular categories.

Uses ordered rule-based classification (first match wins).
Rules are ordered most-specific-first to prevent ambiguity.
"""

import pandas as pd
import re


# ---------------------------------------------------------------------------
# Helpers: detect claimant-side vs respondent-side authorship
# ---------------------------------------------------------------------------

# Country/state names that typically appear as respondent
_RESPONDENT_STATES = [
    "argentina", "canada", "mexico", "ecuador", "venezuela", "bolivia",
    "colombia", "peru", "chile", "guatemala", "el_salvador", "honduras",
    "costa_rica", "panama", "uruguay", "nicaragua", "dominican_republic",
    "egypt", "india", "pakistan", "kazakhstan", "turkmenistan", "uzbekistan",
    "russia", "ukraine", "romania", "poland", "czech", "slovakia", "hungary",
    "croatia", "serbia", "albania", "moldova", "lithuania", "latvia",
    "spain", "italy", "germany", "australia", "mongolia", "indonesia",
    "sri_lanka", "congo", "burundi", "zimbabwe", "tanzania", "kenya",
    "nigeria", "libya", "tunisia", "morocco", "armenia", "georgia",
    "norway", "korea", "cyprus", "belize", "mauritius", "singapore",
]

# Patterns that suggest enforcement/petition actions (claimant-side even when state-named)
_ENFORCEMENT_KEYWORDS = [
    "petition_to_recognize", "petition_to_enforce", "petition_to_confirm",
    "motion_for_recognition", "motion_for_enforcement", "application_for_enforcement",
    "application_to_confirm", "request_for_recognition",
]


def _is_claimant_side(name: str) -> bool:
    """Check if doc_name indicates claimant-side authorship."""
    claimant_markers = [
        "claimant", "claimants", "claiment",  # includes misspelling
        "cliamants",  # misspelling
        "investor_s_", "investor_s ",
        "investors_", "petitioner_s_", "petitioner_s ",
        "petitioners_", "plaintiff_s_", "plaintiff_s ",
        "plaintiffs_", "applicant_s_", "applicants_",
    ]
    for m in claimant_markers:
        if m in name:
            return True
    return False


def _is_respondent_side(name: str) -> bool:
    """Check if doc_name indicates respondent-side authorship."""
    # Explicit respondent markers
    respondent_markers = [
        "respondent", "respondents", "defendant_s_", "defendants_",
        "government_of_canada", "government_of_canada_s",
    ]
    for m in respondent_markers:
        if m in name:
            return True

    # Country-named submissions (e.g., canada_s_counter_memorial)
    for state in _RESPONDENT_STATES:
        if f"{state}_s_" in name:
            # Check it's not an enforcement action BY the state (which would be claimant-side)
            if any(kw in name for kw in _ENFORCEMENT_KEYWORDS):
                return False
            return True

    return False


# ---------------------------------------------------------------------------
# Court-related detection
# ---------------------------------------------------------------------------

_COURT_INDICATORS = [
    "court", "tribunal_de_grande", "bundesgerichtshof", "bundesverfassungsgericht",
    "kammergericht", "landgericht", "oberlandesgericht", "hovr_tt",
    "tingsr_tt", "cour_de_cassation", "cjeu", "ecj",
    "magistrates_court", "high_court", "supreme_court", "district_court",
    "court_of_appeal", "court_of_cassation", "federal_court",
    "superior_court", "commercial_court",
]


def _classify_redacted(name: str) -> str:
    """Classify a redacted_* doc by its underlying type."""
    if "respondent" in name:
        if "statement_of_costs" in name:
            return "respondent_submission_on_costs"
        return "respondent_other"
    if "claimant" in name:
        return "claimant_other"
    if "memorial" in name:
        return "claimant_memorial"
    if "rejoinder" in name:
        return "respondent_rejoinder"
    if "reply" in name:
        return "claimant_reply"
    if "statement_of_claim" in name:
        return "claimant_statement_of_claim"
    if "request_for_arbitration" in name:
        return "notice_of_arbitration"
    if "hearing" in name:
        return "hearing_transcript"
    if "tribunal" in name:
        return "decision_other"
    if "procedural" in name:
        return "procedural_order"
    return "other"


def _is_court_doc(name: str) -> bool:
    """Check if doc_name refers to a domestic/international court document."""
    return any(ind in name for ind in _COURT_INDICATORS)


# ---------------------------------------------------------------------------
# Main classification function
# ---------------------------------------------------------------------------

def classify(name: str) -> str:
    """Apply ordered rules to classify a doc_name string.

    Rules are applied most-specific-first; the first match wins.
    """
    n = name.lower().strip()

    # ===== 1. HEARING TRANSCRIPTS (before party submissions) =====
    if re.search(r"hearing.*transcript|transcript.*hearing|hearing.*transcri|oral_hearing_transcript|hearing_trancript", n):
        return "hearing_transcript"
    if n in ("hearing_transcripts", "combined_hearing_transcripts", "hearings_of_13_14_15_and_16_april_2015"):
        return "hearing_transcript"
    if re.search(r"^(hearing_transcript|hearing_transcripts|hearing_transcription)", n):
        return "hearing_transcript"
    if re.search(r"^(day_\d+_hearing_transcript|closing_arguments_day_\d+_transcript)", n):
        return "hearing_transcript"
    if re.search(r"^(first_procedural_hearing_transcript|annulment_proceeding_transcript)", n):
        return "hearing_transcript"
    if re.search(r"hearing_on_.*transcript|hearing_on_.*day_\d+", n):
        return "hearing_transcript"
    # Standalone transcript patterns (no "hearing" keyword)
    if re.search(r"^transcript_day_\d+|^transcript_of_", n):
        return "hearing_transcript"

    # ===== 2. INDIVIDUAL ARBITRATOR OPINIONS (before awards/decisions) =====

    # Concurring AND dissenting
    if re.search(r"concurring_and_dissenting|dissenting_and_concurring|concurrent_and_dissent", n):
        return "concurring_and_dissenting_opinion"

    # Partial dissenting
    if re.search(r"partial_dissent|partially_dissenting|partial_dissenting", n):
        return "partial_dissenting_opinion"

    # Dissenting (general)
    if re.search(r"dissent(?:ing)?_opinion|dissent(?:ing)?_vote|^dissent_|_dissent$|dissent_to_|dissent_on_|dissent_regarding|statement_of_dissent|note_of_dissent|opini.*n_disidente|dissenting_opinon", n):
        return "dissenting_opinion"

    # Concurring
    if re.search(r"concurring_opinion|concurring_statement|individual_concurring", n):
        return "concurring_opinion"

    # Separate opinion
    if re.search(r"separate_opinion|individual_opinion|individual_statement|statement_of_individual_opinion", n):
        return "separate_opinion"

    # Declaration by arbitrator (appended to awards/decisions)
    if re.search(r"^declaration_by_(?:professor|arbitrator|president)|^declaration_of_professor|^declaration_appended|^declaration_of_arbitrator|^additional_declaration_by|^declaration_on_costs_of_arbitrator|^declaration_dissenting|^declaracion_of_arbitrator|^declaration_of_philippe_sands$|^declaration_of_piet_eeckhout$|^declaration_of_w_michael_reisman$|^declaration_of_professor_w_michael_reisman$|^declaration_of_professor_marcelo_kohen|^declaration_of_professor_steffen_hindelang$|^declaration_of_stan_brijs$|^declaration_of_franklin_berman|^declaration_of_v_v_veeder", n):
        return "declaration_by_arbitrator"

    # Statement of dissent to procedural orders
    if re.search(r"statement_of_dissent.*procedural_order|dissent.*procedural_order_no", n):
        return "statement_of_dissent"

    # ===== 3. SPECIFIC AWARD SUBTYPES (before general award) =====

    # Correction/rectification of award
    if re.search(r"correction_of.*award|correction_to.*award|corrections_to.*award|rectification_of.*award|rectification_of_the_decision|correction_and_interpretation_of.*award|correction_and_interpretation_of_the_award", n):
        return "correction_of_award"

    # Supplementary decision / addendum to award
    if re.search(r"supplementary_decision|addendum_to.*award|additional_award", n):
        return "supplementary_decision"

    # Award on jurisdiction
    if re.search(r"award_on_jurisdiction|award_on_preliminary_objection|award_on_respondent_s_bifurcated_preliminary|award_on_respondent_s_preliminary_objection|jurisdictional_award|award_on_the_respondent_s_application_under", n):
        return "award_on_jurisdiction"

    # Award on costs
    if re.search(r"award_on_costs|award_in_respect_of_costs|final_award_on_costs|final_award_costs|final_award_regarding_costs|final_award_concerning_the_apportionment_of_costs", n):
        return "award_on_costs"

    # Award on damages/quantum
    if re.search(r"award_on_damages|award_on_quantum|award_in_respect_of_damages", n):
        return "award_on_damages"

    # Award on merits
    if re.search(r"award_on_the_merits|award_on_merits|award_on_jurisdiction_and_merits", n):
        return "award_on_merits"

    # Award on liability
    if re.search(r"award_on_jurisdiction_and_liability|award_on_liability|partial_award_on_jurisdiction_and_liability|partial_award_on_liability", n):
        return "award_on_liability"

    # Consent/settlement award
    if re.search(r"consent_award|settlement_award|award_embodying.*settlement", n):
        return "consent_award"

    # Interim award
    if re.search(r"interim_award|interm_award|emergency_award|first_interim_award|interim_award_on", n):
        return "interim_award"

    # Partial award
    if re.search(r"partial_award|partial_final_award|first_partial_award|second_partial_award|third_partial_award|fourth_partial_award|preliminary_tribunal_awards", n):
        return "partial_award"

    # Resubmission award
    if re.search(r"resubmission_award|award.*resubmission_proceeding", n):
        return "resubmission_award"

    # Award on abuse of process
    if "award_on_abuse_of_process" in n:
        return "award"

    # Award on discontinuation
    if "award_on_discontinuation" in n:
        return "award"

    # Rule 41(5) award
    if re.search(r"award.*rule_41", n):
        return "award_on_jurisdiction"

    # ===== 4. GENERAL AWARD =====
    if re.search(r"^(?:final_)?(?:arbitral_)?award|^arbitration_award|^award(?:_|$)|^final_award|^redacted_award|^final_awards_on|^award_of_the|^award_and_separate|^award_corrected|^award_dispositif|^award_english|^award_excerpts|^award_french|^award_operative|^award_redacted|^award_reproduced|^award_russian|^award_spanish|^award_under|^excerpts_of.*award|^extracts_of_award|^final_arbitral_award", n):
        return "award"

    # ===== 5. DECISION ON ANNULMENT =====
    if re.search(r"decision_on_annulment|decision_on.*application_for_annulment|decision_on_the_application_for_annulment|decision_on_the_argentine_republic_s.*annulment|decision_on_the_annulment_application|decisions_on_annulment|decision_annulment_proceeding|decision_on_the_application_for_partial_annulment|decision_on_admissibility_of_the_application_for_annulment", n):
        return "decision_on_annulment"

    # ===== 6. SPECIFIC DECISION SUBTYPES (before general decision) =====

    # Decision on disqualification/challenge
    if re.search(r"decision_on.*disqualif|decision_on.*challenge|decision_on_the_proposal.*disqualif|decision_on_the_claimant.*disqualif|decision_on_the_respondent.*disqualif|decision_on_the_parties.*disqualif|decision_on_the_proposals_to_disqualif|decision_on_callenges|challenge_decision|disqualification_of|chair.*decision_on.*disqualif|chairman.*decision_on.*disqualif|reasoned_decision_on.*disqualif|recommendation.*disqualif|decision_on_the_claimants_proposal_to_disqualif|proposal_for_disqualification", n):
        return "decision_on_disqualification"

    # Decision on stay of enforcement
    if re.search(r"decision_on.*stay_of_enforcement|decision_on.*stay.*enforcement|decision_on.*continuation.*stay|decision_on_the_request.*stay|decision_on.*request.*continued_stay|decision_on.*termination.*stay|decision.*terminate.*stay|decision_on_lifting|decision_on_stay|decision_on_continued_stay|decision_terminating_the_stay|decision_to_terminate_the_stay|decision_on_suspension_of_the_stay", n):
        return "decision_on_stay_of_enforcement"

    # Decision on provisional/interim measures
    if re.search(r"decision_on.*provisional_measures|decision_on.*interim_measures|decision_on.*interim_relief|provisional_measures_decision|decision_on_the_applicant.*provisional|decision_on_the_claimant.*provisional|decision_on_the_claimant.*interim|decision_on_claimant.*interim|decision_on_claimant.*provisional|decision_on_respondent.*provisional|recommendation_on_provisional_measures|decision_on_request_for_suspension_of_proceedings_and_on_request_for_provisional_measures", n):
        return "decision_on_provisional_measures"

    # Decision on bifurcation
    if re.search(r"decision_on_bifurcation|decision_on.*bifurcated|decision_on_the_bifurcated|decision_on_request_for_bifurcation", n):
        return "decision_on_bifurcation"

    # Decision on costs
    if re.search(r"decision_on_costs|decision_on_arbitration_expenses", n):
        return "decision_on_costs"

    # Decision on security for costs
    if re.search(r"decision_on.*security_for_costs|decision_on_el_salvador.*security", n):
        return "decision_on_security_for_costs"

    # Decision on document production
    if re.search(r"decision_on.*document_production|decision_on.*production_of_documents|decision_on_objections_to_document_production|decision_on_parties_requests_for_production|decisions_on_respondent_requests_for_document", n):
        return "decision_on_document_production"

    # Decision on rectification/correction/interpretation
    if re.search(r"decision_on_rectification|decision_on_correction|decision_on_interpretation|decision_on.*request.*rectification|decision_on_the_request.*interpretation|decision_on_the_request.*correction|decision_on_request_for_correction|decision_on_the_requests_for_correction|decision_on_the_requests_for_rectification|decision_on_romania.*interpretation|decision_on_spain.*rectification|decision_on_claimant.*correction|decision_on_claimant.*rectification|decision_on_claimant.*supplementary|decision_on_claimants.*rectification|decision_on_claimants.*supplementary|decision_regarding_claimant.*correction|decision_on_respondent.*supplementary|decision_on_claimant.*request_for_supplementation|decision_on_the_claimant.*rectification|decision_on_request_for_rectification|decision_on_the_request.*joinder", n):
        return "decision_on_rectification"

    # Decision on early dismissal (Rule 41(5))
    if re.search(r"decision.*rule_41|decision_on.*respondent.*rule_41|decision_on_the_respondent.*rule_41|decision_on_the_claimants.*rule_41|decision_on.*respondent_s_objection_under.*rule|decision_on_respondent_s_application_to_dismiss|decision_on_the_admissibility_of_respondent_s_preliminary_objection_to_the_jurisdiction.*rule_41|rule_41_5_decision|decision_under_rule_41", n):
        return "decision_on_early_dismissal"

    # Decision on non-disputing party / amicus
    if re.search(r"decision_on.*non_disputing|decision_on.*amicus|decision_on.*amici|decision_on_authority_to_accept_amicus|decision_on_application_and_submission_by_quechan|decision_on_minbyun|decision_on_the_eu_commission.*intervene|decision_on_the_european_commission.*intervenor|decision_on_the_application.*intervene|decision_on_the_application.*non_disputing|decision_on_motion_to_add_a_new_party|decision_on_the_non_disputing", n):
        return "decision_on_non_disputing_party"

    # Decision on consolidation
    if re.search(r"decision_on_consolidation|decision_on_the_constitution", n):
        return "decision_on_consolidation"

    # Decision on place/seat of arbitration
    if re.search(r"decision_on.*place_of_arbitration|decision_on.*seat_of_arbitration|decision_on_the_seat|decision_on_venue|decision_on_the_place", n):
        return "decision_on_place_of_arbitration"

    # Decision on reconsideration
    if re.search(r"decision_on.*reconsideration|decision_on_ecuador.*reconsideration|decision_on_the_italian.*reconsideration|decision_on_the_kingdom.*reconsideration|decision_on_respondent_request_for_reconsideration|decision_on_reconsideration_and_award", n):
        return "decision_on_reconsideration"

    # Decision on liability
    if re.search(r"decision_on_liability|decision_on_responsibility|decision_on_remaining_issues", n):
        return "decision_on_liability"

    # Decision on merits
    if re.search(r"decision_on_(?:the_)?merits", n):
        return "decision_on_merits"

    # Decision on jurisdiction
    if re.search(r"decision_on_jurisdiction|decision_on_objections_to_jurisdiction|decision_on_preliminary_objection|decision_on_preliminary_issues|decision_on_preliminary_question|decision_on.*jurisdictional|decision_on_hearing_of_respondent_s_objection_to_competence|decision_on_the_objection_to_jurisdiction|decision_on_objection_to_jurisdiction|decision_on_expedited.*objection|decision_on_intra_eu|decision_on_the_achmea|decision_on_the_intra_eu|decision_on_croatia|decision_on_inter_state|decision_on_article|decision_on_respondent.*preliminary|decision_on_claimant.*preliminary|decision_on_the_respondent.*preliminary|decision_on_the_claimant.*preliminary|decision_on_track|decision_on_admissibility|decision_on_the_admissibility|decision_on_termination_request_and_intra_eu|further_decision_on_objections|second_decision_on_objections", n):
        return "decision_on_jurisdiction"

    # Decision on revision
    if re.search(r"decision_on_revision|decision_on_application_for_revision|decision_on_set_aside_of_revision|decision_on_application_to_dismiss_the_revision", n):
        return "decision_on_revision"

    # Decision on representation / waiver
    if re.search(r"decision_on.*representation|decision_on.*waiver|decision_on_applicant.*representation|decision_on_application_for_disqualification_of_counsel|decision_on_the_respondent.*application_for_waiver", n):
        return "decision_other"

    # Decision on enforcement from ICSID/tribunal level
    if re.search(r"decision_on_the_enforcement|decision_on_enforceability", n):
        return "decision_other"

    # Decision on claimant application for additional award / request
    if re.search(r"decision_on_claimant_application_for_additional_award|decision_on_claimant.*request_of|decision_on_claimant.*request_to_dismiss|decision_on_claimants_request.*partial_award|decision_on_claimant.*application_for_dismissal|decision_on_claimant.*motion", n):
        return "decision_other"

    # Generic "decision_on_" that didn't match above
    if re.search(r"^decision_on_", n):
        return "decision_other"

    # ===== 7. APPLICATION FOR ANNULMENT =====
    if re.search(r"application_for_annulment|application_for_partial_annulment|request_for_annulment|request_for_the_annulment|claimants_request_for_the_annulment", n):
        return "application_for_annulment"

    # ===== 8. ANNULMENT MEMORIALS =====
    if re.search(r"memorial.*annulment|annulment_memorial|memorial_in_support_of.*annulment|reply_memorial_in_support_of.*annulment|counter_memorial_on_annulment|reply_on_annulment|rejoinder_on.*annulment|response_to.*application_for_annulment|response_to_the_application_for_annulment|counter_memorial.*annulment|memorial_concerning.*annulment", n):
        return "annulment_memorial"

    # ===== 9. AMICUS CURIAE =====
    if re.search(r"amicus_curiae|amicus_brief|amicus_submission|amici_curiae|application.*amicus|petition_for_amicus|petition.*amicus_curiae|application_for_leave_to_file_an_amicus|application_for_leave_to_make_amicus|application_for_permission_to_proceed_as_amici|receipt_confirmation_of_petition_for_amicus|invitation_to_amici|new_amicus_petition|brief_amicus_curiae|request_to_file_a_written_submission_amicus", n):
        return "amicus_curiae"

    # ===== 10. NON-DISPUTING PARTY SUBMISSIONS =====
    if re.search(r"non_disputing_party|article_1128_submission|1128_submission|nafta_article_1128|non_disputing_state_party", n):
        return "non_disputing_party_submission"

    # ===== 11. CLAIMANT SUBMISSIONS (check claimant-side first for party-specific docs) =====
    if _is_claimant_side(n):
        # Claimant press release
        if "press_release" in n:
            return "claimant_press_release"
        # Claimant letter
        if re.search(r"letter|e_mail|correspondence", n):
            return "claimant_letter"
        # Claimant statement of claim
        if re.search(r"statement_of_claim|notice_of_arbitration_and_statement_of_claim|notice_of_demand.*statement_of_claim|particularized_statement_of_claim|amended_statement_of_claim|revised_statement_of_claim|second_amended.*statement_of_claim|third_amended.*statement_of_claim|complaint", n):
            return "claimant_statement_of_claim"
        # Claimant post-hearing
        if re.search(r"post_hearing|closing_memorial|closing_argument|opening_statement", n):
            return "claimant_post_hearing"
        # Claimant reply
        if re.search(r"reply(?!.*annulment)|rebuttal|sur_reply|surrejoinder", n):
            return "claimant_reply"
        # Claimant counter-memorial (on jurisdiction, annulment)
        if "counter_memorial" in n:
            return "claimant_counter_memorial"
        # Claimant rejoinder
        if "rejoinder" in n:
            return "claimant_rejoinder"
        # Claimant memorial
        if "memorial" in n:
            return "claimant_memorial"
        # Claimant submission on costs
        if re.search(r"submission_on_costs|cost_submission|statement_of_costs|costs_submission|submission_on_costs|reply_costs|reply.*cost|reply_statement_of_costs", n):
            return "claimant_submission_on_costs"
        # Claimant observations
        if re.search(r"observation|comments_on|submission_on|submission_re|submission_regarding|submission_pursuant|submission_challenging|submission_to|submissions_on|opposition|response_to|answer", n):
            return "claimant_observations"
        # Claimant request (interim measures, document production, etc.)
        if re.search(r"request_for|application_for|motion_for|motion_to|petition|challenge_to|proposal.*disqualif|waiver|notice_of_additional|notice_of_change|notice_of_challenge", n):
            return "claimant_request"
        # Claimant witness / expert evidence
        if re.search(r"witness_statement|expert_report|expert_statement|book_of_expert|book_of_witness", n):
            return "expert_report"
        # Claimant stock exchange filing
        if "stock_exchange" in n:
            return "stock_exchange_filing"
        # Claimant notice of arbitration
        if re.search(r"notice_of_arbitration|notice_of_intent|notice_of_dispute", n):
            return "notice_of_arbitration"
        # Claimant other (fallback for claimant-side docs)
        return "claimant_other"

    # ===== 12. RESPONDENT SUBMISSIONS =====
    if _is_respondent_side(n):
        # Respondent press release
        if "press_release" in n:
            return "respondent_press_release"
        # Respondent letter
        if re.search(r"letter|e_mail|correspondence|notice_of_supplemental|notice_of_authority", n):
            return "respondent_letter"
        # Respondent statement of defence
        if re.search(r"statement_of_defen[cs]e|statement_of_defence", n):
            return "respondent_statement_of_defence"
        # Respondent preliminary objection
        if re.search(r"preliminary_objection|objection.*jurisdiction|expedited_objection|notification_of_preliminary", n):
            return "respondent_preliminary_objection"
        # Respondent request for bifurcation
        if re.search(r"request_for_bifurcation|application_for_bifurcation|renewed_request_for_bifurcation|bifurcation_request", n):
            return "respondent_request_for_bifurcation"
        # Respondent post-hearing
        if re.search(r"post_hearing|closing_statement|opening_statement|presentation_for", n):
            return "respondent_post_hearing"
        # Respondent reply
        if re.search(r"(?<!counter_)reply(?!.*annulment)|rebuttal|sur_reply", n):
            return "respondent_reply"
        # Respondent counter-memorial
        if "counter_memorial" in n:
            return "respondent_counter_memorial"
        # Respondent rejoinder
        if "rejoinder" in n:
            return "respondent_rejoinder"
        # Respondent memorial on jurisdiction
        if re.search(r"memorial_on_jurisdiction|memorial_on_preliminary|memorial_on_objection|memorial_on_admissibility|memorial_on_waiver|memorial_on_article|memorial_on_costs", n):
            return "respondent_memorial_on_jurisdiction"
        # Respondent submission on costs
        if re.search(r"submission_on_costs|cost_submission|statement_of_costs|costs_submission|request_for_costs", n):
            return "respondent_submission_on_costs"
        # Respondent observations
        if re.search(r"observation|comments_on|submission_on|submission_re|submission_regarding|submission_pursuant|submission_to|submissions_on|response_to|answer|summary_of", n):
            return "respondent_observations"
        # Respondent request (security for costs, provisional measures, etc.)
        if re.search(r"request_for|application_for|motion_for|motion_to|petition|challenge_to|proposal.*disqualif|objection|emergency_motion|remedy_claimed|summons", n):
            return "respondent_request"
        # Respondent notice
        if re.search(r"response_to.*notice|response_to_the_notice|response.*notice_of_arbitration", n):
            return "respondent_observations"
        # Respondent other (fallback)
        return "respondent_other"

    # ===== 13. DOMESTIC COURT DOCUMENTS =====

    # Check for court judgments, decisions, orders
    if _is_court_doc(n) or re.search(r"^judgment_|^judgement_|^decision_of_the_|^decision_of_|^order_of_the_.*court|^order_of_the_.*district|^order_of_us_|^order_of_the_us_|^order_of_the_united_states|^order_us_|^opinion_of_the_us_|^opinion_of_us_|^opinion_of_the_united|^memorandum_opinion|^memorandum_and_order|^memorandum_order|^memorandum_decision|^opinion_and_order|^opinion_order|^report_and_recommendation", n):

        # Memorandum opinion (US courts)
        if re.search(r"memorandum_opinion", n):
            return "court_memorandum_opinion"

        # US court opinion
        if re.search(r"^opinion_of_.*(?:us_|united_states_|u_s_).*(?:court|circuit|district)|^opinion_of_us_|^opinion_us_court|^opinion_united_states_court|^opinion_of_the_us_|^opinion_of_the_united_states", n):
            return "court_opinion"

        # Court opinion (other)
        if re.search(r"^opinion_of_.*court|^opinion_and_order", n):
            return "court_opinion"

        # Enforcement proceedings
        if re.search(r"petition_to_enforce|petition_to_confirm|petition_to_recognize|petition_for_confirmation|petition_for_recognition|application_to_confirm|application_for_enforcement|cross_petition_for_confirmation|enforcement_judgment|enforcement_decision|enforcement_order|enforcement_proceeding|confirmation_of_arbitral|exequatur|endorsement.*recognition|endorsement.*enforcement|motion_for_recognition|motion_for_enforcement|demand_for_payment|register.*award|memo_endorsement|default_judgment|motion_for_default|petition_for_an_order_confirming", n):
            return "enforcement_proceeding"

        # Set-aside proceedings
        if re.search(r"set_aside|vacate|annul.*award|challenge.*award|application_to_set_aside|recours_en_annulation|non_justiciability_of_challenge", n):
            return "set_aside_proceeding"

        # Court judgment
        if re.search(r"judgment|judgement", n):
            return "court_judgment"

        # Court decision
        if re.search(r"^decision_of_|decision_from_|decision_by_|^decision_of", n):
            return "court_decision"

        # Court order
        if re.search(r"order_of_.*court|order_of_.*district|^order_and_final_judgment|^order_and_judgment|^order_from_|^order_of_the", n):
            return "court_order"

        # Court memorandum/report
        if re.search(r"memorandum_and_order|memorandum_order|memorandum_decision|report_and_recommendation|opinion_and_order|opinion_order", n):
            return "court_memorandum_opinion"

        # Fallback for court docs
        return "court_other"

    # Additional enforcement/set-aside docs not caught above
    if re.search(r"petition_to_enforce|petition_to_confirm|petition_to_recognize|petition_for_confirmation|petition_for_recognition|application_to_confirm|application_for_enforcement|enforcement_judgment|enforcement_decision|enforcement_order|exequatur|^petition_to_us_district|^petition_to_the_us_|register.*enforce.*award|^petition_to_vacate|^petition_for_an_order_confirming", n):
        return "enforcement_proceeding"
    if re.search(r"set_aside|vacate.*award|application_to_set_aside|challenge.*arbitral_award|challenge_to_arbitral|recours_en_annulation|non_justiciability_of_challenge", n):
        return "set_aside_proceeding"

    # Court briefs (filed in domestic courts)
    if re.search(r"^brief_(?:for_|of_|in_|amicus)|^brief_and_special|^final_brief_for|^proposed_brief", n):
        return "court_brief"

    # Court motions
    if re.search(r"^motion_(?:for_|to_|of_)|^consolidated_response_in_opposition|^consolidated_sur_reply|^opposition_to_petition|^opposition_to_claimant|^opposition_to_petitioner|^reply_brief_for|^reply_in_support_of|^reply_memorandum_in|^reply_to_united_states_opposition|^memorandum_in_(?:opposition|support)|^memorandum_of_law|^memorandum_of_points|^petitioner_s_|^petitioners_|^answer_to_complaint|^answer_affirmative", n):
        return "court_motion"

    # Declarations filed in court proceedings
    if re.search(r"^declaration_of_(?!.*arbitrat).*(?:in_support|petition|motion|affidavit|court|district|judgment|proceeding)", n):
        return "affidavit"

    # Additional court-related judgments/decisions/opinions
    if re.search(r"^judgment_of_|^judgement_of_|^judgment_by_|^judgment_from_|^decision_of_(?:the_)?(?:amsterdam|antwerp|brussels|paris|london|hague|singapore|ontario|quebec|svea|swiss|stockholm|swedish|dutch|english|french|german|austrian|belgian|uk_|us_|delhi|kiev|kyiv|ukraine|russian|madrid|rome|lisbon|luxembourg|cayman|caribbean|belize|new_zealand|mauritius|australian|federal_|supreme_|high_|district_|ontario_|superior_|commercial_)", n):
        return "court_judgment"

    if re.search(r"^decision_of_the_ad_hoc_committee", n):
        return "decision_on_annulment"

    if re.search(r"^decision_of_the_arbitral_tribunal|^decision_of_the_tribunal|^decision_by_tribunal|^decision_by_the_tribunal|^decision_by_the_court_of_arbitration|^decision_for_termination|^decision_french|^decision_and_order_by_the_arbitral", n):
        return "decision_other"

    # Remaining standalone court decisions (catch-all for decision_of_the_*court*)
    if re.search(r"^decision_of_(?:the_)?(?:amsterdam|antwerp|brussels|paris|london|hague|singapore|ontario|quebec|svea|swiss|stockholm|swedish|dutch|english|french|german|austrian|belgian|uk_|us_|delhi|kiev|kyiv|ukraine|russian|madrid|rome|lisbon|luxembourg|nacka|pechersk|specialized|federal_|supreme_|high_|district_|superior_|commercial_|eu_|european_)", n):
        return "court_decision"

    # ===== 14. EXPERT EVIDENCE =====
    if re.search(r"expert_report|expert_opinion|expert_declaration|expert_witness_report|expert_witness_statement|expert_statement|expert_valuation|expert_reply_report|rebuttal_expert_report|reply_expert_report|rejoinder_expert_report|response_expert_report|compass_lexecon|nera_expert|brattle_expert|first_expert_opinion|first_legal_opinion|legal_opinion_of_|legal_opinion_by_|legal_opinion_on_|legal_opinion_prepared|legal_opinion_submitted|opinion_of_(?:professor|prof_|sir_|dr_|pierre|philippe|jan_paulsson|w_michael|anne_marie|christopher_greenwood|richard_b|adnan|eduardo|jose|sompong|thomas_w|claus|don_wallace|fernando|ove_bring|domingo|alejandro)", n):
        # Distinguish legal opinions from expert reports
        if re.search(r"legal_opinion", n):
            return "legal_opinion"
        if re.search(r"^opinion_of_(?!.*court|.*us_|.*united_states|.*district|.*circuit|.*appeal|.*advocate)", n):
            return "legal_opinion"
        return "expert_report"

    # Witness statements
    if re.search(r"witness_statement|affidavit|affirmation_of|first_affidavit|second_affidavit|direct_testimony|rebuttal_witness|rejoinder_witness|reply_witness|first_witness|second_witness|third_witness", n):
        return "witness_statement"

    # ===== 15. PROCEDURAL ORDERS =====
    if re.search(r"procedural_order|procedural_oder|procedural_resolution|first_procedural_order|draft_procedural_order|procedural_orders_no|procedural_no_1|clarification_to_procedural_order|amended_procedural_order|revised_procedural_order|redacted_procedural_order", n):
        return "procedural_order"

    # ===== 16. NOTICE DOCUMENTS =====
    # Notice of arbitration
    if re.search(r"notice_of_arbitration|notice_of_and_request_for_arbitration|notice_of_demand_for_arbitration|notice_of_claim|notice_of_the_commencement|request_for_arbitration|request_for_institution|request_for_access_to_the_additional_facility|request_for_approval_of_arbitration|request_for_institution_of_arbitration|amended_notice_of_arbitration|redacted_request_for_arbitration|amended_request_for_arbitration|notice_of_arbitration_to_submit|cover_letter_to_notice_of_arbitration|preliminary_response_to_the_notice|response_to_notice_of_arbitration|response_to_the_notice_of_arbitration|claimants_request_for_arbitration|claimants_amended_request|notice_of_commencement", n):
        return "notice_of_arbitration"

    # Notice of intent
    if re.search(r"notice_of_intent|notice_of_intention|final_notice_of_dispute", n):
        return "notice_of_intent"

    # Notice of dispute
    if re.search(r"notice_of.*dispute|notice_of_an_investment_dispute|notice_of_investment_dispute|notice_of_legal_dispute|notice_of_violations", n):
        return "notice_of_dispute"

    # Other notices
    if re.search(r"notice_of_discontinuance|notice_of_errata|notice_of_appeal|notice_of_removal|notice_of_satisfaction|notice_of_withdrawal|notice_of_voluntary|notice_of_hearing|notice_of_case_manager|notice_of_emergency|notice_of_motion|notice_of_petition|notice_of_proposed|notice_of_appearance|notice_of_supplemental_authority", n):
        return "notice_other"

    # ===== 17. SETTLEMENT =====
    if re.search(r"settlement_agreement|settlement_deed|annex_a_settlement", n):
        return "settlement_agreement"

    # ===== 18. DISCONTINUANCE ORDERS =====
    if re.search(r"discontinuance_order|order.*discontinu|order_taking_note_of_the_discontinuance|order_of.*discontinuance|cover_letter_order_of_the_secretary_general_taking_note|order_on_discontinuance|order_terminating_arbitral|order_for_the_termination|order_of_the_committee.*discontinu|order_of_the_tribunal.*discontinu|decision_for_termination|joint_order_on_the_costs_of_arbitration_and_for_the_termination", n):
        return "discontinuance_order"

    # ===== 19. CONFIDENTIALITY ORDERS =====
    if re.search(r"confidentiality_order|confidentiality_agreement|amended_confidentiality|confidentiality_protocol", n):
        return "confidentiality_order"

    # ===== 20. ADMINISTRATIVE / INSTITUTIONAL =====

    # ICSID communications
    if re.search(r"^icsid_|^icsidletter|^letter_from_icsid|^letter_from_the_icsid|^letter_to_icsid|^icsid_s_|registration_notification|first_icsid_letter|order_of_the_secretary_general|cover_letter_order_of_the_secretary_general", n):
        return "icsid_communication"

    # PCA communications
    if re.search(r"^pca_(?!press_release)|^letter_from_the_pca|^letter_from_pca|^letter_to_amici_from_permanent_court|pca_s_second_letter", n):
        return "pca_communication"

    # PCA press releases
    if re.search(r"^pca_press_release", n):
        return "pca_communication"

    # Tribunal communications
    if re.search(r"^letter_from_the_tribunal|^letter_from_tribunal|^letter_to_the_parties|^letter_to_parties|^direction_of_the_tribunal|^directions_from_the_tribunal|^directions_concerning|^letter_regarding|^letter_extension|^letter_request_for_extension|^letter_on_disclosure|^letter_appointing|^notification_to_non_disputing", n):
        return "tribunal_communication"

    # Minutes
    if re.search(r"^minutes_|^final_minutes", n):
        return "minutes"

    # Terms of appointment
    if re.search(r"terms_of_appointment|terms_of_reference|rules_of_procedure|agreement_on_certain_procedural|procedural_calendar|amended_procedural_calendar|revised_procedural_calendar|amended_procedural_timetable|revised_procedural_timetable|procedural_timetable|revised_timetable|revised_annex|amended_procedural_oder", n):
        return "terms_of_appointment"

    # ===== 21. PRESS RELEASES (non-party-specific) =====
    if re.search(r"press_release|news_release", n):
        return "press_release"

    # ===== 22. GENERAL DECISIONS (tribunal/committee, not court) =====
    if re.search(r"^decision_|^interim_decision|^emergency_decision|^first_decision_on|^second_decision_on|^preliminary_decision|^final_decision|^reasoning_of_the_decision|^reasons_for_the_tribunal|^reasons_for_decision", n):
        return "decision_other"

    # ===== 23. REMAINING SPECIFIC CATEGORIES =====

    # Counter-memorials (unattributed)
    if re.search(r"^counter_memorial", n):
        return "respondent_counter_memorial"

    # Memorials (unattributed)
    if re.search(r"^memorial_of_the_investor|^memorial_of_the_investors|^memorial_on_the_merits|^memorial_on_jurisdiction|^memorial_on_objections|^memorial_on_annulment|^memorial_on_compliance|^memorial_in_support|^memorial_of_claimants|^memorial_of_consolidation|^memorial_on_jurisdictional|^memorial_on_the_merits", n):
        # Try to figure out if claimant or respondent
        if re.search(r"investor|claimant|applicant", n):
            return "claimant_memorial"
        if re.search(r"counter_memorial|objections_to_jurisdiction|respondent", n):
            return "respondent_memorial_on_jurisdiction"
        return "claimant_memorial"

    # Replies / rejoinders / rebuttals (unattributed)
    if re.search(r"^reply_on_annulment|^reply_on_the_merits|^reply_on_jurisdiction|^reply_on_provisional|^reply_on_objections|^reply_memorial|^reply_and_joinder|^rejoinder_on_|^rejoinder_of_the|^rejoinder_memorial|^rebuttal_memorial|^rejoinder_to_|^reply_counter_memorial|^reply_of_claimants|^reply_of_respondent|^reply_on_jurisdictional|^reply_to_chile|^reply_to_claimant|^reply_to_united_states|^reply_in_support|^response_to_application|^response_to_chile|^response_to_canada|^response_to_claimant|^response_on_provisional|^response_to_motion|^response_to_respondent|^response_to_notice|^response_to_the_republic|^response_to_the_notice|^reply_brief_for|^reply_to_|^response_from_|^response_of_|^response_on_|^response_expert|^response_to_the_|^rebuttal_report|^rebuttal_witness", n):
        # Attribute based on content
        if re.search(r"of_the_investor|of_claimant|claimant|investor|applicant|petitioner", n):
            return "claimant_reply"
        if re.search(r"of_the_respondent|of_respondent|of_the_u_s|of_the_government|of_canada|respondent|united_states_of_america_on", n):
            return "respondent_reply"
        if re.search(r"expert|witness|rebuttal_report|rebuttal_witness", n):
            return "expert_report"
        return "claimant_reply"

    # Post-hearing submissions (unattributed)
    if re.search(r"^post_hearing", n):
        return "respondent_post_hearing"  # typically respondent when unattributed

    # Stock exchange filings
    if "stock_exchange" in n:
        return "stock_exchange_filing"

    # Errata
    if re.search(r"^errata", n):
        return "errata"

    # Index of exhibits
    if re.search(r"index_of_(?:factual_)?exhibit|index_of_legal_authorit|cumulative_index|list_of_.*exhibit|list_of_.*authorit|updated_table_of_authorit|list_of_documents_on_pacer|list_of_parties_expert|appendices_to", n):
        return "index_of_exhibits"

    # Annexes
    if re.search(r"^annex(?:_|es_)|^annex$", n):
        return "annex"

    # Declarations filed in court (affidavits)
    if re.search(r"^declaration_of_|^affidavit_of_|^affirmation_of_", n):
        return "affidavit"

    # Order (generic, non-court — from tribunal/committee)
    if re.search(r"^order(?:_|$)", n):
        return "procedural_order"

    # EU Commission documents
    if re.search(r"european_commission|eu_commission", n):
        return "non_disputing_party_submission"

    # Joint submissions / stipulations
    if re.search(r"^joint_|^disputing_parties_agreement|^parties_joint", n):
        return "other"

    # Letters (from various entities, not caught above)
    if re.search(r"^letter_(?:from_|to_|by_|of_|regarding|on_disclosure)|^email_from_|^e_mail_from_", n):
        return "tribunal_communication"

    # Legal opinions (not caught above)
    if re.search(r"^legal_opinion|^opinion_of_|^opinion_on_|^opinion_with_respect|^opinion_dissidente|^conclusions_of_|^comments_relating_to", n):
        return "legal_opinion"

    # Court opinions (advocate general, etc.)
    if re.search(r"^opinion_of_advocate_general|^opinion_of_the_advocate|^opinion_of_the_attorney_general|^opinion_of_the_procurator|^opinion_of_the_state_attorney|^dutch_supreme_court.*advocate|^conclusions_of_advocate", n):
        return "court_opinion"

    # Remaining court-related documents
    if re.search(r"^judgment_|^judgement_|^ruling_|^reasons_for_judgment|^reasons_for_order|^reasons_for_decision|^grounds_of_decision|^certification|^certiorari|^decree_of|^resolution_of|^default_judgment|^restraining_notice|^interim_charging_order", n):
        return "court_other"

    # Remaining opinions from US courts
    if re.search(r"^opinion_of_court|^opinion_of_delaware|^opinion_us_court|^opinion_united_states", n):
        return "court_opinion"

    # Remaining decisions from courts
    if re.search(r"^decision_of_", n):
        return "court_decision"

    # Revision application
    if re.search(r"revision_application|revision_decision|application_for_revision|request_for_correction|request_for_supplementary_decision|request_for_supplementation|request_for_the_disqualification|request_for_disqualification", n):
        return "decision_on_rectification"

    # Recommendation by third party re disqualification
    if re.search(r"^recommendation_", n):
        return "decision_on_disqualification"

    # Privilege logs, reports
    if re.search(r"privilege_master|privilege_log|report_on_inadvertent", n):
        return "other"

    # Complaints (various)
    if re.search(r"^complaint", n):
        return "court_motion"

    # Interim orders/rulings from tribunal
    if re.search(r"^interim_order|^interim_ruling|^interim_decision|^order_for_interim|^order_no_|^order_on_", n):
        return "procedural_order"

    # Remaining hearing-related
    if re.search(r"^hearing_on_|^oral_hearing|^first_session|^first_procedural_hearing", n):
        return "hearing_transcript"

    # ===== NAMED PARTY SUBMISSIONS (company/entity names as claimant) =====
    # These are specific named claimants (not prefixed with claimant_/investor_)
    _named_claimant_patterns = [
        r"^glamis_", r"^canfor_", r"^tembec_", r"^grand_river_",
        r"^methanex_s_", r"^mercer_international", r"^pac_rim_cayman",
        r"^dibc_s_", r"^loewen_s_", r"^gami_s_", r"^metalclad_s_",
        r"^rand_s_", r"^raymond_loewen_s_", r"^caratube_",
        r"^renco_", r"^9ren_s_", r"^nextera_s_", r"^air_india_s_",
        r"^almaden_s_", r"^agility_s_", r"^ascom_s_", r"^berkeley_s_",
        r"^cerro_verde_s_", r"^adf_", r"^misen_s_", r"^omega_s_",
        r"^crystallex_s_", r"^gold_reserve_s_", r"^holcim_s_",
        r"^deutsche_telekom_s_", r"^merck_s_", r"^saint_gobain_s_",
        r"^eiser_s_", r"^anatolie_s_", r"^naftogaz_", r"^pao_tatneft",
        r"^oi_european", r"^oschadbank_s_", r"^rusoro_mining",
        r"^mol_s_", r"^arnol.*porter", r"^peteris_pildegovics",
        r"^arctic_fishing", r"^gml_s_", r"^chevron_",
    ]
    for pat in _named_claimant_patterns:
        if re.search(pat, n):
            if "press_release" in n:
                return "claimant_press_release"
            if re.search(r"letter|correspondence|e_mail", n):
                return "claimant_letter"
            if "counter_memorial" in n:
                return "claimant_counter_memorial"
            if "rejoinder" in n:
                return "claimant_rejoinder"
            if "reply" in n:
                return "claimant_reply"
            if "memorial" in n:
                return "claimant_memorial"
            if re.search(r"post_hearing|closing|opening", n):
                return "claimant_post_hearing"
            if re.search(r"submission_on_costs|statement_of_costs", n):
                return "claimant_submission_on_costs"
            if re.search(r"notice_of_arbitration|notice_of_intent|notice_of_dispute|notice_regarding", n):
                return "notice_of_arbitration"
            if re.search(r"observation|submission|response|request|motion|opposition|petition|complaint|objection", n):
                return "claimant_observations"
            return "claimant_other"

    # Named respondent-state submissions (without _s_ possessive)
    if re.search(r"^(?:preliminary_objections_of_(?:the_united_states|guatemala))", n):
        return "respondent_preliminary_objection"
    if re.search(r"^(?:repondent_s_)", n):  # misspelling
        return "respondent_rejoinder" if "rejoinder" in n else "respondent_other"
    if re.search(r"^opening_submission_of_the_republic|^presentation_of_uruguay", n):
        return "respondent_post_hearing"

    # Unattributed observations/submissions
    if re.search(r"^(?:additional_)?observations_(?:of_|on_)|^observations_of_judge", n):
        return "respondent_observations"
    if re.search(r"^additional_observations", n):
        return "claimant_observations"

    # Fifth/eighth/fourth/etc US submissions (NDP)
    if re.search(r"^(?:first|second|third|fourth|fifth|sixth|seventh|eighth)_(?:u_s_|submission_of_the_united_states|mexican_|submission_of_canada|u_s_submission)", n):
        return "non_disputing_party_submission"

    # Factum/intervenor submissions
    if re.search(r"^factum_of_the_intervenor|^outline_of_argument_of_intervenor", n):
        return "non_disputing_party_submission"

    # Revised/redacted submissions (unattributed)
    if re.search(r"^redacted_(?:memorial|rejoinder|reply|statement_of_claim|request|respondent|claimant|hearing|tribunal|procedural)", n):
        return _classify_redacted(n)

    # Amended statement of claim (unattributed)
    if re.search(r"^(?:amended_)?statement_of_claim|^second_amended_statement|^revised_statement_of_claim", n):
        return "claimant_statement_of_claim"

    # Court petition/filing patterns not caught above
    if re.search(r"^petition_for_(?:a_writ|rehearing|discovery|limited_participation)|^petition_to_(?:stay|brazilian)|^petition_by_|^ex_parte_|^appeal_(?:from_|of_order)|^appell", n):
        return "court_motion"

    # Court-related: clerk's entry, satisfaction, default
    if re.search(r"^clerk_s_|^satisfaction_of_|^default_of_|^restraining_notice", n):
        return "court_other"

    # Objection to jurisdiction (unattributed)
    if re.search(r"^objection_to_jurisdiction|^objection_to_subpoena|^expedited_preliminary_objection", n):
        return "respondent_preliminary_objection"

    # Request for bifurcation (unattributed - usually respondent)
    if re.search(r"^request_for_bifurcation", n):
        return "respondent_request_for_bifurcation"

    # Application for provisional measures (unattributed)
    if re.search(r"^application_for_provisional_measures|^application_pursuant", n):
        return "claimant_request"

    # Confirmation of arbitral award
    if re.search(r"^confirmation_of_arbitral", n):
        return "enforcement_proceeding"

    # Demand for payment
    if re.search(r"^demand_for_payment|^demand_letter", n):
        return "enforcement_proceeding"

    # Challenge to arbitrator (unattributed)
    if re.search(r"^challenge_to_(?!arbitral)|^pnb_banka_s_disqualification|^disqualification_proposal", n):
        return "decision_on_disqualification"

    # Petition for NDP participation
    if re.search(r"petition_for_participation_as_non|petition_for_limited_participation|national_mining_association|quechan_indian_nation|chamber_of_commerce.*amicus|earth_justice|iids_s_|iisd_s_", n):
        return "amicus_curiae"

    # Disclosure / report docs
    if re.search(r"^disclosure_|^report_on_inadvertent", n):
        return "other"

    # Newspaper/media articles
    if re.search(r"^newspaper_article|^investment_law_and_policy", n):
        return "other"

    # Decrees/contracts
    if re.search(r"^decree_of_|^concession_contract|^presidential_decree|^czech_arnold_porter", n):
        return "other"

    # Notes by arbitrators
    if re.search(r"^note_by_arbitrator|^clarification_of_", n):
        return "separate_opinion"

    # Resignation letters
    if re.search(r"^resignation_|^philippe_sands_letter_of_resignation", n):
        return "other"

    # Requests for stay/suspension/continuation
    if re.search(r"^request_for_continuation|^request_for_suspension|^request_for_correction|^request_for_summons|^request_for_entry|^request_for_leave|^request_for_a_preliminary", n):
        return "claimant_request"

    # Second/additional declarations (court filings)
    if re.search(r"^(?:second|third|fourth)_declaration_of_|^declaration$|^second_affidavit", n):
        return "affidavit"

    # Named state-related enforcement petitions
    if re.search(r"^oman_s_petition|^nicaragua_s_petition|^poland_s_petition|^panama_s_(?:application|petition|motion|memorandum|press_release|counter_memorial|rejoinder|response|submission|request)", n):
        # Distinguish: if it's a petition to enforce (claimant action), route there
        if re.search(r"petition.*enforce|petition.*confirm|petition.*recognize|application_for_writ|motion_for_judgment.*enforce", n):
            return "enforcement_proceeding"
        if "press_release" in n:
            return "respondent_press_release"
        if "counter_memorial" in n:
            return "respondent_counter_memorial"
        if "rejoinder" in n:
            return "respondent_rejoinder"
        if "response" in n:
            return "respondent_observations"
        if "submission_on_costs" in n:
            return "respondent_submission_on_costs"
        if "request" in n:
            return "respondent_request"
        return "respondent_other"

    # More enforcement patterns
    if re.search(r"^cross_petition|^motion_for_default_judgment|^motion_for_issuance_of_a_letter_rogatory|^proposed_order", n):
        return "court_motion"

    # Unattributed reply briefs / memoranda in court proceedings
    if re.search(r"^reply_memorandum_of_law_of_the_republic|^reply_in_support_of_(?:bulgaria|the_republic)", n):
        return "court_motion"

    # Export report (misspelling of expert)
    if re.search(r"^export_report", n):
        return "expert_report"

    # Questions from tribunal
    if re.search(r"^questions_and_claimant", n):
        return "claimant_observations"

    # Joint motions/submissions
    if re.search(r"^joint_(?:consent_motion|motion|status_report|stipulation|request)", n):
        return "court_motion"

    # Jurisdiction of the arbitral tribunal (published excerpts)
    if re.search(r"^jurisdiction_of_the_arbitral", n):
        return "decision_on_jurisdiction"

    # Action in annulment
    if re.search(r"^action_in_annulment", n):
        return "set_aside_proceeding"

    # Remaining respondent-related patterns
    if re.search(r"^maduro_government|^interim_president", n):
        return "court_motion"

    # Mexican/US NDP submissions
    if re.search(r"^mexico_(?:article_1128|second_article|submission_on_the_bilcon)", n):
        return "non_disputing_party_submission"

    # Canada observations
    if re.search(r"^canada_observations", n):
        return "non_disputing_party_submission"

    # Unattributed post-hearing
    if re.search(r"^post_(?:hearing|argument)_", n):
        return "claimant_post_hearing"

    # Addendum to expert report / privilege report
    if re.search(r"^addendum_to_(?:expert|the_privilege)", n):
        return "expert_report"

    # EU/CJEU decisions (not caught above)
    if re.search(r"^cjeu_|^court_of_justice_of_the_european|^eu_commission_press|^european_commission_(?:consultation|decision|letter|observations|press)", n):
        return "court_decision" if re.search(r"judgment|decision", n) else "press_release"

    # Uncategorized numbered docs
    if re.search(r"^\d+$", n):
        return "other"

    # ===== U.S. AS RESPONDENT (u_s_ and us_ prefixed docs) =====
    if re.search(r"^u_s_|^us_", n):
        # NDP submissions first (article references)
        if re.search(r"article_1128|article_10_20|article_10_19|article_1120|nafta_ft[ca]|submission_re_nafta|ndp_submission|submission_on_the_bilcon", n):
            return "non_disputing_party_submission"
        # Court doc
        if re.search(r"memorandum_decision|memorandum_and_order", n):
            return "court_memorandum_opinion"
        # Respondent submissions
        if "counter_memorial" in n:
            return "respondent_counter_memorial"
        if re.search(r"statement_of_defen[cs]e", n):
            return "respondent_statement_of_defence"
        if "objection_to_jurisdiction" in n:
            return "respondent_preliminary_objection"
        if "rejoinder" in n:
            return "respondent_rejoinder"
        if re.search(r"reply", n):
            return "respondent_reply"
        if "memorial" in n:
            return "respondent_memorial_on_jurisdiction"
        if re.search(r"post_hearing", n):
            return "respondent_post_hearing"
        if re.search(r"submission_on_costs|comments_on.*costs", n):
            return "respondent_submission_on_costs"
        if "request_for_bifurcation" in n:
            return "respondent_request_for_bifurcation"
        if re.search(r"request_for_consolidation|request_for_.*hearing|motion_to_", n):
            return "respondent_request"
        if "letter" in n:
            return "respondent_letter"
        if re.search(r"observation|submission_on|response_to|submission_in_support|final_observations|submission_pursuant|submission_on_place", n):
            return "respondent_observations"
        if re.search(r"amended_statement|supplemental_statement", n):
            return "respondent_statement_of_defence"
        return "respondent_other"

    # united_states_* patterns
    if re.search(r"^united_states_", n):
        if re.search(r"article_|submission_on_the_bilcon|written_submission_pursuant", n):
            return "non_disputing_party_submission"
        if "letter" in n:
            return "respondent_letter"
        return "non_disputing_party_submission"

    # the_united_states_of_america_s_memorial
    if re.search(r"^the_united_states", n):
        if "memorial" in n:
            return "respondent_memorial_on_jurisdiction"
        return "respondent_other"

    # ===== TRIBUNAL PATTERNS (not caught above) =====
    if re.search(r"^tribunal_e_mail|^tribunal_letter|^tribunal_s_(?:communication|letter|procedural|order|consolidation)|^tribunal_questions|^tribunal_decisions", n):
        if re.search(r"order_on_|consolidation_order|order_english", n):
            return "procedural_order"
        return "tribunal_communication"

    # ===== UNATTRIBUTED STATEMENT OF DEFENCE/DEFENSE =====
    if re.search(r"^statement_of_defen[cs]e", n):
        return "respondent_statement_of_defence"

    # ===== STATEMENT_OF_* PATTERNS =====
    if re.search(r"^statement_of_rejoinder", n):
        return "respondent_rejoinder"
    if re.search(r"^statement_of_reply", n):
        return "claimant_reply"
    if re.search(r"^statement_of_particulars", n):
        return "claimant_statement_of_claim"
    if re.search(r"^statement_of_material_facts|^statement_of_points_and_authorities", n):
        return "court_motion"
    if re.search(r"^statement_of_montenegro", n):
        return "respondent_observations"
    if re.search(r"^statement_of_appellant", n):
        return "court_motion"
    if re.search(r"^statement_of_(?:armis|hector|jorge|marcelo|stephan|w_joel|jack)", n):
        return "witness_statement"

    # ===== SUBMISSION_OF_* / SUBMISSION_BY_* PATTERNS =====
    if re.search(r"^submission_of_|^submission_by_|^submissions_of_the_united", n):
        if re.search(r"amici|sierra_club|canadian_union|postal_workers|steelworkers|earthworks", n):
            return "amicus_curiae"
        if re.search(r"non_disputing|article_1128", n):
            return "non_disputing_party_submission"
        if re.search(r"canada|mexico|el_salvador|peru|united_states|government_of", n):
            return "non_disputing_party_submission"
        if re.search(r"moldavia|apotex", n):
            return "claimant_request"
        return "non_disputing_party_submission"

    # ===== SUPPLEMENTAL_* / SUPPLEMENTARY_* PATTERNS =====
    if re.search(r"^supplemental_", n):
        if "award" in n:
            return "supplementary_decision"
        if "counter_memorial" in n:
            return "respondent_counter_memorial"
        if "memorial" in n:
            return "claimant_memorial"
        if "brief" in n:
            return "court_motion"
        if "declaration" in n:
            return "affidavit"
        if re.search(r"statement_of_(?!defen)", n):
            return "witness_statement"
        if "submission" in n:
            return "non_disputing_party_submission"
        return "other"
    if re.search(r"^supplementary_", n):
        if "award" in n:
            return "supplementary_decision"
        if "reasons_for_judgment" in n:
            return "court_judgment"
        return "other"

    # ===== NAMED CLAIMANTS NOT CAUGHT ABOVE =====
    if re.search(r"^tennant_energy|^infrastructure_services|^tlgi_", n):
        if "counter_memorial" in n:
            return "claimant_counter_memorial"
        if "rejoinder" in n:
            return "claimant_rejoinder"
        if re.search(r"post_hearing", n):
            return "claimant_post_hearing"
        if re.search(r"submission|written_submission|outline", n):
            return "claimant_observations"
        return "claimant_other"

    # track_2_ (Ecuador respondent submissions)
    if re.search(r"^track_2_", n):
        if "counter_memorial" in n:
            return "respondent_counter_memorial"
        return "respondent_rejoinder"

    # the_republic_s_ (respondent submissions)
    if re.search(r"^the_republic_s_", n):
        return "respondent_counter_memorial"

    # ===== SEPARATE OPINIONS / DECLARATIONS BY ARBITRATORS =====
    if re.search(r"^separate_declaration_of_|^separate_statement_of_", n):
        return "separate_opinion"
    if re.search(r"^separate_dissenting_", n):
        return "dissenting_opinion"

    # ===== WITNESS / EXPERT PATTERNS NOT CAUGHT ABOVE =====
    if re.search(r"^witness_declaration|^witness_statment|^witness_and_legal_statement|^direct_presentation", n):
        return "witness_statement"
    if re.search(r"^second_expert_|^supplement_to_the_privilege", n):
        return "expert_report"

    # ===== FRENCH / SPANISH AWARDS AND DECISIONS =====
    if re.search(r"^sentence_finale|^laudo_|^sentencia_", n):
        return "award"
    if re.search(r"^sentence_sur_les_d_clinatoires", n):
        return "decision_on_jurisdiction"
    if re.search(r"^section_41_5_decision", n):
        return "decision_on_early_dismissal"

    # ===== COURT-RELATED PATTERNS =====
    if re.search(r"^stipulated_order|^stipulation_and_order", n):
        return "court_order"
    if re.search(r"^temporary_restraining_order", n):
        return "court_order"
    if re.search(r"^summons_in_a_civil", n):
        return "court_other"
    if re.search(r"^writ_of_summons", n):
        return "court_other"
    if re.search(r"^special_master", n):
        return "court_other"
    if re.search(r"^notice_as_to_request|^notice_of_russian", n):
        return "notice_other"
    if re.search(r"^appelees_motion|^reply_brief_of_appellant|^response_to_contempt|^rosneft_s_motion|^viroel_micula|^alixpartners|^application_by_chevron|^buttonwood|^acf_s_complaint|^joint_(?:consent_motion|status_report|stipulation)", n):
        return "court_motion"
    if re.search(r"^\d+_ewhc_|^\d+_ewca_", n):
        return "court_judgment"
    if re.search(r"^supplementary_reasons_for_judgment", n):
        return "court_judgment"

    # Swiss federal tribunal decisions
    if re.search(r"^swiss_federal_tribunal", n):
        return "court_decision"
    # Stipulated protective order
    if re.search(r"^stipulated_protective_order", n):
        return "confidentiality_order"

    # ===== ENFORCEMENT / STAY =====
    if re.search(r"^stay_of_enforcement", n):
        return "decision_on_stay_of_enforcement"
    if re.search(r"^stay_of_proceedings", n):
        return "procedural_order"
    if re.search(r"^settlement_of_the_unpaid", n):
        return "enforcement_proceeding"

    # ===== DISCONTINUANCE / TERMINATION =====
    if re.search(r"^termination_order", n):
        return "discontinuance_order"
    if re.search(r"^third_partial_and_final_award", n):
        return "partial_award"

    # ===== ADMINISTRATIVE / ICSID =====
    if re.search(r"^second_icsid|^additional_request_from_icsid|^the_secretary_general_of_icsid", n):
        return "icsid_communication"
    if re.search(r"^updated_hearing_schedule", n):
        return "terms_of_appointment"

    # ===== AMICUS / NDP =====
    if re.search(r"^sierra_club|^petition_of_the_canadian_union|^response_by_the_canadian_union|^revised_submission_of_non_disputing", n):
        return "amicus_curiae"
    if re.search(r"^second_submission_of_the_republic", n):
        return "non_disputing_party_submission"

    # ===== RESPONDENT PATTERNS (misc) =====
    if re.search(r"^argentine_republic_s_", n):
        return "respondent_observations"
    if re.search(r"^grounds_of_the_proposal", n):
        return "respondent_request"
    if re.search(r"^pdvsa_s_", n):
        return "respondent_observations"
    if re.search(r"^comunicaci_n_de_la_rep", n):
        return "respondent_letter"
    if re.search(r"^spain_supplementary", n):
        return "respondent_observations"
    if re.search(r"^venezuela_dpsva|^venezuela_and_gold_reserve", n):
        return "respondent_request" if "motion" in n else "press_release"
    if re.search(r"^d_plica", n):
        return "respondent_rejoinder"

    # ===== CLAIMANT PATTERNS (misc) =====
    if re.search(r"^second_post_hearing_submission_of_the_investor", n):
        return "claimant_post_hearing"
    if re.search(r"^plaintiff_", n):
        return "claimant_letter"
    if re.search(r"^supplement_to_claim", n):
        return "claimant_memorial"
    if re.search(r"^requ_te_en_vue", n):
        return "claimant_request"

    # ===== TRIBUNAL / ARBITRATOR LETTERS =====
    if re.search(r"^judge_.*letter|^lord_phillips_recommendation|^prof_.*letter|^professor_stern_s_comments|^archer_s_letter", n):
        return "tribunal_communication"
    if re.search(r"^seventh_declaration_of_professor", n):
        return "declaration_by_arbitrator"

    # ===== PRESS RELEASES =====
    if re.search(r"^foreign_affairs_and_international|^gold_reserve_enters", n):
        return "press_release"

    # ===== MISC =====
    if re.search(r"^request_for_consultation", n):
        return "notice_of_dispute"
    if re.search(r"^status_report$", n):
        return "other"
    if re.search(r"^summary_minutes", n):
        return "minutes"
    if re.search(r"^summary_of_award", n):
        return "award"

    # ===== FALLBACK =====
    return "other"


# ---------------------------------------------------------------------------
# Broad document category mapping (granular_category -> doc_category)
# ---------------------------------------------------------------------------

_DOC_CATEGORY_MAP = {
    # Awards
    "award": "Awards",
    "award_on_jurisdiction": "Awards",
    "award_on_costs": "Awards",
    "award_on_damages": "Awards",
    "award_on_merits": "Awards",
    "award_on_liability": "Awards",
    "partial_award": "Awards",
    "interim_award": "Awards",
    "consent_award": "Awards",
    "correction_of_award": "Awards",
    "supplementary_decision": "Awards",
    "resubmission_award": "Awards",
    # Decisions
    "decision_on_jurisdiction": "Decisions",
    "decision_on_annulment": "Decisions",
    "decision_on_liability": "Decisions",
    "decision_on_merits": "Decisions",
    "decision_on_provisional_measures": "Decisions",
    "decision_on_stay_of_enforcement": "Decisions",
    "decision_on_bifurcation": "Decisions",
    "decision_on_costs": "Decisions",
    "decision_on_disqualification": "Decisions",
    "decision_on_security_for_costs": "Decisions",
    "decision_on_document_production": "Decisions",
    "decision_on_rectification": "Decisions",
    "decision_on_early_dismissal": "Decisions",
    "decision_on_non_disputing_party": "Decisions",
    "decision_on_consolidation": "Decisions",
    "decision_on_place_of_arbitration": "Decisions",
    "decision_on_reconsideration": "Decisions",
    "decision_on_revision": "Decisions",
    "decision_other": "Decisions",
    # Procedural Orders
    "procedural_order": "Procedural Orders",
    # Arbitrator Opinions
    "dissenting_opinion": "Arbitrator Opinions",
    "concurring_opinion": "Arbitrator Opinions",
    "concurring_and_dissenting_opinion": "Arbitrator Opinions",
    "separate_opinion": "Arbitrator Opinions",
    "partial_dissenting_opinion": "Arbitrator Opinions",
    "declaration_by_arbitrator": "Arbitrator Opinions",
    "statement_of_dissent": "Arbitrator Opinions",
    # Claimant Submissions
    "claimant_memorial": "Claimant Submissions",
    "claimant_reply": "Claimant Submissions",
    "claimant_counter_memorial": "Claimant Submissions",
    "claimant_rejoinder": "Claimant Submissions",
    "claimant_post_hearing": "Claimant Submissions",
    "claimant_statement_of_claim": "Claimant Submissions",
    "claimant_submission_on_costs": "Claimant Submissions",
    "claimant_observations": "Claimant Submissions",
    "claimant_request": "Claimant Submissions",
    "claimant_letter": "Claimant Submissions",
    "claimant_press_release": "Claimant Submissions",
    "claimant_other": "Claimant Submissions",
    # Respondent Submissions
    "respondent_counter_memorial": "Respondent Submissions",
    "respondent_memorial_on_jurisdiction": "Respondent Submissions",
    "respondent_rejoinder": "Respondent Submissions",
    "respondent_reply": "Respondent Submissions",
    "respondent_post_hearing": "Respondent Submissions",
    "respondent_statement_of_defence": "Respondent Submissions",
    "respondent_preliminary_objection": "Respondent Submissions",
    "respondent_request_for_bifurcation": "Respondent Submissions",
    "respondent_submission_on_costs": "Respondent Submissions",
    "respondent_observations": "Respondent Submissions",
    "respondent_request": "Respondent Submissions",
    "respondent_letter": "Respondent Submissions",
    "respondent_press_release": "Respondent Submissions",
    "respondent_other": "Respondent Submissions",
    # Notice Documents
    "notice_of_arbitration": "Notice Documents",
    "notice_of_intent": "Notice Documents",
    "notice_of_dispute": "Notice Documents",
    "notice_other": "Notice Documents",
    # Domestic Court Proceedings
    "court_judgment": "Domestic Court Proceedings",
    "court_decision": "Domestic Court Proceedings",
    "court_order": "Domestic Court Proceedings",
    "court_memorandum_opinion": "Domestic Court Proceedings",
    "court_opinion": "Domestic Court Proceedings",
    "enforcement_proceeding": "Domestic Court Proceedings",
    "set_aside_proceeding": "Domestic Court Proceedings",
    "court_brief": "Domestic Court Proceedings",
    "court_motion": "Domestic Court Proceedings",
    "court_other": "Domestic Court Proceedings",
    # Third-Party Submissions
    "amicus_curiae": "Third-Party Submissions",
    "non_disputing_party_submission": "Third-Party Submissions",
    # Expert Evidence
    "expert_report": "Expert Evidence",
    "witness_statement": "Expert Evidence",
    # Hearing Transcripts
    "hearing_transcript": "Hearing Transcripts",
    # Annulment Proceedings
    "application_for_annulment": "Annulment Proceedings",
    "annulment_memorial": "Annulment Proceedings",
    # Administrative Documents
    "icsid_communication": "Administrative Documents",
    "pca_communication": "Administrative Documents",
    "tribunal_communication": "Administrative Documents",
    "minutes": "Administrative Documents",
    "terms_of_appointment": "Administrative Documents",
    "confidentiality_order": "Administrative Documents",
    # Settlement Agreements
    "settlement_agreement": "Settlement Agreements",
    # Discontinuance Orders
    "discontinuance_order": "Discontinuance Orders",
    # Other (Press Releases, Other Document Types, Miscellaneous)
    "press_release": "Other",
    "legal_opinion": "Other",
    "affidavit": "Other",
    "stock_exchange_filing": "Other",
    "errata": "Other",
    "index_of_exhibits": "Other",
    "annex": "Other",
    "other": "Other",
}


def get_doc_category(granular_category: str) -> str:
    """Map a granular_category to its broad doc_category."""
    return _DOC_CATEGORY_MAP.get(granular_category, "Other")


# ---------------------------------------------------------------------------
# Main: Load, classify, save
# ---------------------------------------------------------------------------

def main():
    df = pd.read_csv("data/lookup_table.csv")
    df["granular_category"] = df["doc_name"].apply(classify)
    df["doc_category"] = df["granular_category"].map(get_doc_category)
    df.to_csv("data/lookup_table.csv", index=False)

    # Print summary
    vc = df["granular_category"].value_counts()
    print(f"\nTotal rows: {len(df)}")
    print(f"Unique doc_names: {df['doc_name'].nunique()}")
    print(f"Granular categories: {vc.shape[0]}")
    if "other" in vc.index:
        other_count = vc["other"]
        print(f"Granular 'other' count: {other_count} ({other_count/len(df)*100:.1f}%)")
    print(f"\n--- Granular Category Distribution ---")
    print(vc.to_string())

    dc = df["doc_category"].value_counts()
    print(f"\n--- Doc Category Distribution ---")
    print(dc.to_string())


if __name__ == "__main__":
    main()
