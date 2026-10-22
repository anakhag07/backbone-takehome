from copy import deepcopy

from src.clinical import _adherence, build_abstraction, calculate_metrics, interval_minutes


def interval(start, end):
    return {"start": start, "end": end}


def claim(**updates):
    base = {
        "patient_id": "patient-1", "encounter_id": "visit-1", "appointment_id": None,
        "service_date": "2026-01-19", "service_type": "individual",
        "record_kind": "clinical_note", "record_status": "final", "attendance": "attended",
        "patient_present": True, "contact_intervals": [], "presence_intervals": [],
        "therapy_intervals": [], "excluded_intervals": [], "reported_minutes": None,
        "corrects_fields": [], "corrects_document_id": None, "is_copy": False,
        "evidence": ["Example evidence"], "narrative": "", "uncertainty": None,
        "doc_id": "doc-1", "source_doc_id": "source-1",
    }
    base.update(updates)
    return base


def abstraction(claims, plans=None, assessments=None):
    return build_abstraction({"patient_id": "patient-1"}, [{"encounters": claims, "plans": plans or [], "assessments": assessments or [], "observations": [], "warnings": []}])


def plan(**updates):
    item = {"effective_start": "2026-01-05", "effective_end": None, "episode_start": "2026-01-05", "episode_end": "2026-01-30", "min_days": 3, "min_minutes": 150, "week_start": "monday", "eligible_services": ["individual", "group", "family"], "patient_present_required": True, "doc_id": "plan-1", "evidence": ["The goal"]}
    item.update(updates)
    return item


def test_intervals_measure_union_intersection_and_exclusion():
    result = interval_minutes([interval("10:00", "11:00"), interval("10:30", "11:30")], [interval("10:45", "11:00")], [interval("10:15", "11:15")])
    assert result == 45


def test_telehealth_segments_are_one_session():
    result = abstraction([claim(contact_intervals=[interval("13:00", "13:20"), interval("13:30", "13:55")])])
    metrics = result["metrics"]["overall"]
    assert metrics["sessions"]["value"] == 1
    assert metrics["minutes"]["value"] == 45


def test_scoped_correction_survives_later_stale_copy():
    original = claim(service_type="group", record_kind="attendance", presence_intervals=[interval("10:00", "11:30")])
    content = claim(service_type="group", doc_id="content", therapy_intervals=[interval("10:00", "11:30")], excluded_intervals=[interval("10:45", "11:00")])
    correction = claim(service_type="group", doc_id="correction", record_kind="correction", presence_intervals=[interval("10:00", "11:15")], corrects_fields=["departure"], corrects_document_id="source-1")
    copy = {**original, "doc_id": "later-copy", "record_kind": "copy", "is_copy": True}
    result = abstraction([original, content, correction, copy])
    assert result["metrics"]["overall"]["minutes"]["value"] == 60
    assert result["metrics"]["overall"]["sessions"]["value"] == 1
    assert "Applied explicit correction to presence_intervals" in result["events"][0]["decisions"]


def test_conflicting_whole_contact_accounts_remain_alternatives():
    result = abstraction([
        claim(contact_intervals=[interval("09:00", "09:50")], reported_minutes=50),
        claim(doc_id="doc-2", contact_intervals=[interval("09:10", "09:50")], reported_minutes=40),
    ])
    metrics = result["metrics"]["overall"]
    assert metrics["sessions"]["value"] == 1
    assert metrics["minutes"]["value"] is None
    assert metrics["minutes"]["alternatives"] == [40, 50]


def test_duplicate_claims_and_input_order_do_not_change_clinical_results():
    original = claim(reported_minutes=45)
    other = claim(encounter_id="visit-2", service_date="2026-01-20", reported_minutes=30, doc_id="doc-2")
    expected = abstraction([original, other])["metrics"]
    assert abstraction([other, original, deepcopy(original)])["metrics"] == expected


def test_family_patient_presence_clips_provider_session():
    result = abstraction([claim(service_type="family", therapy_intervals=[interval("13:00", "13:45")], presence_intervals=[interval("13:15", "13:45")])])
    assert result["metrics"]["overall"]["minutes"]["value"] == 30


def test_collateral_and_medication_do_not_count_as_therapy():
    result = abstraction([claim(service_type="family", patient_present=False, reported_minutes=30), claim(service_type="medication", encounter_id="med-1", reported_minutes=20)])
    assert result["metrics"]["overall"]["sessions"]["value"] == 0
    assert result["metrics"]["overall"]["minutes"]["value"] == 0


def test_final_absence_overrides_draft_and_charge():
    final = claim(record_kind="attendance", attendance="absent", patient_present=False)
    draft = claim(doc_id="draft", record_kind="draft", record_status="draft", attendance="attended", reported_minutes=90)
    charge = claim(doc_id="charge", record_kind="charge", attendance="attended", reported_minutes=90)
    result = abstraction([draft, charge, final])
    assert result["metrics"]["overall"]["sessions"]["value"] == 0


def test_draft_alone_does_not_establish_delivered_care():
    result = abstraction([claim(record_kind="draft", record_status="draft", reported_minutes=90)])
    metrics = result["metrics"]["overall"]
    assert metrics["sessions"]["alternatives"] == [0, 1]
    assert metrics["minutes"]["max"] is None


def test_unknown_minutes_are_not_silently_zero():
    result = abstraction([claim()])
    metrics = result["metrics"]["overall"]
    assert metrics["sessions"]["value"] == 1
    assert metrics["minutes"]["value"] is None
    assert metrics["minutes"]["min"] == 0
    assert metrics["minutes"]["max"] is None


def test_joint_goal_uses_correlated_scenarios():
    # Two retained complete histories both fail despite favorable marginal maxima.
    metrics = {"scenarios": [
        {"therapy_days_min": 3, "therapy_days_max": 3, "minutes_min": 140, "minutes_max": 140},
        {"therapy_days_min": 2, "therapy_days_max": 2, "minutes_min": 160, "minutes_max": 160},
    ]}
    assert _adherence(metrics, plan())["status"] == "not_met"


def test_goal_uncertainty_and_partial_final_week():
    claims = [claim(encounter_id=f"visit-{i}", service_date=f"2026-01-{day}", reported_minutes=50) for i, day in enumerate((26, 27, 28))]
    claims.append(claim(encounter_id="visit-2", service_date="2026-01-28", reported_minutes=40, doc_id="conflict"))
    result = abstraction(claims, [plan()])
    last = result["metrics"]["weeks"][-1]
    assert last["partial_review_week"] is True
    assert last["plan_adherence"]["status"] == "cannot_determine"


def test_source_form_duplicates_do_not_create_new_measurement():
    item = {"source_form_id": "form-1", "instrument": "scale", "completed_date": "2026-01-16", "score": 14, "interpretation": "Some change", "doc_id": "a", "evidence": ["Result 14"]}
    result = abstraction([], assessments=[item, {**item, "doc_id": "b"}])
    assert len(result["assessments"]) == 1
    assert result["assessments"][0]["source_doc_ids"] == ["a", "b"]


def test_appointment_identifier_links_note_without_encounter_identifier():
    result = abstraction([claim(appointment_id="appointment", reported_minutes=45), claim(encounter_id=None, appointment_id="appointment", doc_id="b", reported_minutes=45)])
    assert len(result["events"]) == 1


def test_midweek_plan_change_is_not_prorated():
    result = abstraction([claim(reported_minutes=60)], [plan(), plan(effective_start="2026-01-21", min_minutes=90, doc_id="new-plan")])
    week = next(w for w in result["metrics"]["weeks"] if w["week_start"] == "2026-01-19")
    assert week["plan_adherence"]["status"] == "cannot_determine"
    assert "proration" in week["plan_adherence"]["reason"]


def test_correlated_contact_and_presence_do_not_create_zero_duration():
    first = claim(contact_intervals=[interval("09:00", "09:50")], presence_intervals=[interval("09:00", "09:50")])
    second = claim(doc_id="alternative", contact_intervals=[interval("10:00", "10:40")], presence_intervals=[interval("10:00", "10:40")])
    result = abstraction([first, second])
    assert result["metrics"]["overall"]["minutes"]["alternatives"] == [40, 50]


def test_distinct_overlapping_events_do_not_claim_exact_sum():
    result = abstraction([claim(contact_intervals=[interval("09:00", "09:50")]), claim(encounter_id="visit-2", contact_intervals=[interval("09:30", "10:00")])])
    metrics = result["metrics"]["overall"]
    assert metrics["sessions"]["value"] == 2
    assert metrics["minutes"]["value"] is None
    assert metrics["overlapping_event_pairs"]


def test_undated_event_is_not_definitely_in_requested_window():
    events = abstraction([claim(service_date=None, reported_minutes=45)])["events"]
    metrics = calculate_metrics(events, [], "2026-01-19", "2026-01-25")
    assert metrics["overall"]["sessions"]["alternatives"] == [0, 1]
    assert metrics["overall"]["minutes"]["alternatives"] == [0, 45]


def test_unidentified_records_do_not_establish_distinct_event_count():
    result = abstraction([claim(encounter_id=None, reported_minutes=45), claim(encounter_id=None, reported_minutes=40, doc_id="b")])
    assert result["metrics"]["overall"]["sessions"]["value"] is None
    assert result["metrics"]["overall"]["sessions"]["min"] == 1
    assert result["metrics"]["overall"]["sessions"]["max"] == 2


def test_one_form_with_conflicting_dates_remains_one_uncertain_assessment():
    item = {"source_form_id": "form-1", "instrument": "scale", "completed_date": "2026-01-16", "score": 14, "interpretation": "", "doc_id": "a", "evidence": ["Result 14"]}
    result = abstraction([], assessments=[item, {**item, "doc_id": "b", "completed_date": "2026-01-26"}])
    assert len(result["assessments"]) == 1
    assert result["assessments"][0]["completed_date"] is None


def test_source_specific_duration_correction_retains_independent_account():
    a = claim(doc_id="a", source_doc_id="source-a", reported_minutes=50)
    b = claim(doc_id="b", source_doc_id="source-b", reported_minutes=40)
    correction = claim(doc_id="c", source_doc_id="source-c", record_kind="correction", reported_minutes=45, corrects_fields=["reported_minutes"], corrects_document_id="source-a")
    result = abstraction([a, b, correction])
    assert result["metrics"]["overall"]["minutes"]["alternatives"] == [40, 45]


def test_departure_correction_preserves_original_arrival():
    original = claim(service_type="group", record_kind="attendance", presence_intervals=[interval("10:00", "11:30")])
    activity = claim(service_type="group", doc_id="activity", therapy_intervals=[interval("10:00", "11:30")])
    correction = claim(service_type="group", doc_id="correction", record_kind="correction", presence_intervals=[interval("10:15", "11:15")], corrects_fields=["departure"], corrects_document_id="source-1")
    result = abstraction([original, activity, correction])
    assert result["metrics"]["overall"]["minutes"]["value"] == 75


def test_arrival_correction_preserves_original_departure():
    original = claim(service_type="group", record_kind="attendance", presence_intervals=[interval("10:00", "11:30")])
    activity = claim(service_type="group", doc_id="activity", source_doc_id="activity-source", therapy_intervals=[interval("10:00", "11:30")])
    correction = claim(service_type="group", doc_id="correction", source_doc_id="correction-source", record_kind="correction", presence_intervals=[interval("10:15", "11:15")], corrects_fields=["arrival"], corrects_document_id="source-1")
    result = abstraction([original, activity, correction])
    assert result["metrics"]["overall"]["minutes"]["value"] == 75


def test_competing_corrections_to_same_source_remain_unresolved():
    original = claim(reported_minutes=50)
    corrections = [claim(doc_id=f"correction-{value}", source_doc_id=f"source-{value}", record_kind="correction", reported_minutes=value, corrects_fields=["reported_minutes"], corrects_document_id="source-1") for value in (40, 45)]
    result = abstraction([original, *corrections])
    assert result["metrics"]["overall"]["minutes"]["alternatives"] == [40, 45]


def test_single_day_window_cannot_establish_full_week_failure():
    claims = [claim(encounter_id=f"visit-{day}", service_date=f"2026-01-{day}", reported_minutes=50) for day in (19, 20, 21)]
    result = abstraction(claims, [plan()])
    scoped = calculate_metrics(result["events"], result["plans"], "2026-01-19", "2026-01-19")
    assert scoped["overall"]["minutes"]["value"] == 50
    assert scoped["weeks"][0]["plan_adherence"]["status"] == "cannot_determine"


def test_full_final_episode_week_can_establish_failure():
    result = abstraction([claim(service_date="2026-01-26", reported_minutes=40)], [plan()])
    last = result["metrics"]["weeks"][-1]
    assert last["partial_review_week"] is True
    assert last["plan_adherence"]["status"] == "not_met"


def test_copy_with_unique_conflicting_value_is_not_discarded():
    result = abstraction([claim(reported_minutes=50), claim(doc_id="copied-source", source_doc_id="copy-of-independent-source", record_kind="copy", is_copy=True, reported_minutes=40)])
    assert result["metrics"]["overall"]["minutes"]["alternatives"] == [40, 50]
