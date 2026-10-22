"""Deterministic care-event reconciliation and arithmetic over documented evidence.

The finite alternatives below describe the available record, not undocumented care.
No model is called here, and no patient facts or question IDs are encoded in rules.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
from hashlib import sha256
from itertools import product
import json
import math


THERAPY_SERVICES = {"individual", "group", "family"}
SCENARIO_LIMIT = 4096


def _unique(values):
    found = {}
    for value in values:
        found[json.dumps(value, sort_keys=True, default=str)] = value
    return [found[key] for key in sorted(found)]


def _iso(value):
    if not value:
        return None
    return date.fromisoformat(value).isoformat()


def _clock(value):
    parts = str(value).split(":")
    if len(parts) != 2:
        raise ValueError(f"Expected HH:MM time, got {value!r}")
    hours, minutes = map(int, parts)
    if not 0 <= hours <= 23 or not 0 <= minutes <= 59:
        raise ValueError(f"Invalid clock time: {value!r}")
    return hours * 60 + minutes


def _intervals(values):
    intervals = []
    for item in values or []:
        start, end = _clock(item["start"]), _clock(item["end"])
        if end <= start:
            raise ValueError("Intervals must have positive duration within one service date")
        intervals.append((start, end))
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _intersection(left, right):
    return [(max(a, c), min(b, d)) for a, b in left for c, d in right if max(a, c) < min(b, d)]


def _subtract(intervals, excluded):
    remaining = list(intervals)
    for left, right in excluded:
        pieces = []
        for start, end in remaining:
            if right <= start or left >= end:
                pieces.append((start, end))
            else:
                if start < left:
                    pieces.append((start, left))
                if right < end:
                    pieces.append((right, end))
        remaining = pieces
    return remaining


def interval_minutes(contact, excluded=None, presence=None):
    """Measure union(contact) intersect presence, minus excluded intervals."""
    result = _intervals(contact)
    if presence is not None:
        result = _intersection(result, _intervals(presence))
    result = _subtract(result, _intervals(excluded))
    return sum(end - start for start, end in result)


def _sources(claims):
    return sorted({c["doc_id"] for c in claims if c.get("doc_id")})


def _evidence(claims):
    return _unique([
        {"doc_id": c.get("doc_id"), "source_doc_id": c.get("source_doc_id"), "quote": quote}
        for c in claims for quote in c.get("evidence", [])
    ])


def _corrects(claim, field):
    if claim.get("record_kind") != "correction":
        return False
    aliases = {
        "presence_intervals": {"arrival", "departure", "patient_arrival", "patient_departure", "arrival_time", "departure_time", "presence_intervals"},
        "reported_minutes": {"reported_minutes", "actual_minutes", "duration", "minutes"},
    }
    return bool(set(claim.get("corrects_fields", [])) & aliases.get(field, {field}))


def _field_claims(claims, field):
    candidates = [c for c in claims if c.get(field) not in (None, "unknown", [], "")]
    if field == "attendance":
        candidates = [c for c in candidates if c.get("record_kind") not in {"draft", "charge"} and c.get("record_status") != "draft" and (c.get("record_kind") != "schedule" or c.get("attendance") in {"absent", "cancelled"})]
    if field in {"contact_intervals", "presence_intervals", "therapy_intervals", "excluded_intervals", "reported_minutes", "patient_present"}:
        candidates = [c for c in candidates if c.get("record_kind") not in {"schedule", "draft", "charge"} and c.get("record_status") != "draft"]
    corrections = [c for c in candidates if _corrects(c, field)]
    if corrections:
        # A correction is field-scoped. An attached old copy never defeats it.
        return corrections
    noncopies = [c for c in candidates if not c.get("is_copy") and c.get("record_kind") != "copy"]
    if noncopies:
        candidates = noncopies
    finals = [c for c in candidates if c.get("record_status") == "final" and c.get("record_kind") not in {"draft", "charge", "schedule"}]
    if finals:
        candidates = finals
    return candidates


def _field_values(claims, field):
    return _unique([c[field] for c in _field_claims(claims, field)])


def _duration_options(claims, warnings):
    minutes = []
    time_options = []
    contacts = _field_values(claims, "contact_intervals")
    presences = _field_values(claims, "presence_intervals")
    activities = _field_values(claims, "therapy_intervals")
    breaks = _field_values(claims, "excluded_intervals") or [[]]
    try:
        # Each complete interval account is an alternative. Never union two accounts.
        for claim in _field_claims(claims, "contact_intervals"):
            scoped_presence = presences if any(_corrects(c, "presence_intervals") for c in claims) else ([claim["presence_intervals"]] if claim.get("presence_intervals") else presences)
            scoped_breaks = breaks if any(_corrects(c, "excluded_intervals") for c in claims) else ([claim["excluded_intervals"]] if claim.get("excluded_intervals") else breaks)
            for presence, excluded in product(scoped_presence or [None], scoped_breaks):
                intervals = _intervals(claim["contact_intervals"])
                if presence is not None:
                    intervals = _intersection(intervals, _intervals(presence))
                intervals = _subtract(intervals, _intervals(excluded))
                minutes.append(sum(end - start for start, end in intervals))
                time_options.append(intervals)
        for presence, activity, excluded in product(presences, activities, breaks):
            intervals = _subtract(_intersection(_intervals(activity), _intervals(presence)), _intervals(excluded))
            minutes.append(sum(end - start for start, end in intervals))
            time_options.append(intervals)
        for value in _field_values(claims, "reported_minutes"):
            number = float(value)
            if not math.isfinite(number) or number < 0:
                raise ValueError("Reported therapy minutes must be finite and nonnegative")
            minutes.append(number)
    except (ValueError, TypeError, KeyError) as exc:
        warnings.append(f"Invalid duration evidence: {exc}")
        return [None], []
    if not minutes:
        warnings.append("Actual patient therapy duration is not established")
        return [None], []
    options = sorted(set(minutes))
    if len(options) > 1:
        warnings.append("Competing supported duration accounts remain unresolved")
    if sum(len(v) > 1 for v in (contacts, presences, activities, breaks)) > 1:
        warnings.append("Cross-field alternatives use conservative combinations; their dependencies need review")
    return options, _unique(time_options)


def _reconcile_event(event_id, claims, uncertain_identity=False):
    warnings, decisions = [], []
    dates = sorted({_iso(c.get("service_date")) for c in claims if c.get("service_date")})
    services = _field_values(claims, "service_type") or ["unknown"]
    known_services = [s for s in services if s != "unknown"]
    services = known_services or ["unknown"]
    if len(dates) != 1:
        warnings.append("Service date is missing or conflicting")
    if len(services) > 1:
        warnings.append("Service type is conflicting")
    if uncertain_identity:
        warnings.append("No shared encounter/appointment identifier; identity may overlap another event")

    attendance = _field_values(claims, "attendance")
    trusted = [c for c in claims if c.get("record_kind") not in {"draft", "schedule", "charge"} and c.get("record_status") != "draft"]
    if attendance and all(v in {"scheduled", "unknown"} for v in attendance):
        attendance = []
    if not attendance and any(c.get("contact_intervals") or (c.get("reported_minutes") or 0) > 0 for c in trusted):
        attendance = ["attended"]
        decisions.append("Actual documented contact supports attendance")
    attended_options = set()
    for value in attendance:
        if value == "attended":
            attended_options.add(True)
        elif value in {"absent", "cancelled"}:
            attended_options.add(False)
        else:
            attended_options.update((False, True))
    if not attended_options:
        attended_options.update((False, True))
        warnings.append("Attendance is not established by a completed-contact record")

    presence = _field_values(claims, "patient_present")
    if not presence and any(c.get("contact_intervals") or c.get("presence_intervals") or (c.get("reported_minutes") or 0) > 0 for c in trusted):
        presence = [True]
    if not presence:
        presence = [False, True]
        if True in attended_options:
            warnings.append("Patient participation is not established")

    minutes, time_options = _duration_options(claims, warnings) if True in attended_options and True in presence and any(s in THERAPY_SERVICES or s in {"unknown", "other"} for s in services) else ([0], [])
    scenarios = []
    for attended, present, service, duration in product(sorted(attended_options), presence, services, minutes):
        eligible_options = [service in THERAPY_SERVICES] if service not in {"unknown", "other"} else [False, True]
        for service_eligible in eligible_options:
            counted = attended and present and service_eligible
            scenarios.append({"sessions": int(counted), "minutes": duration if counted else 0, "service_type": service, "patient_present": present, "attended": attended})
    scenarios = _unique(scenarios)
    for field in ("attendance", "presence_intervals", "reported_minutes", "contact_intervals", "therapy_intervals", "excluded_intervals"):
        if any(_corrects(c, field) for c in claims):
            decisions.append(f"Applied explicit correction to {field}")
    if any(c.get("is_copy") or c.get("record_kind") == "copy" for c in claims):
        decisions.append("Copies retained as evidence without additional encounters or votes")
    if any(c.get("record_status") == "draft" or c.get("record_kind") in {"draft", "charge", "schedule"} for c in claims):
        decisions.append("Scheduling, draft, and billing evidence do not establish delivered therapy")
    if len(attended_options) > 1:
        warnings.append("Attendance has unresolved alternatives")
    return {
        "event_id": event_id,
        "encounter_id": next((c.get("encounter_id") for c in claims if c.get("encounter_id")), None),
        "appointment_ids": sorted({c["appointment_id"] for c in claims if c.get("appointment_id")}),
        "date": dates[0] if len(dates) == 1 else None,
        "date_options": dates,
        "service_type": services[0] if len(services) == 1 else "unknown",
        "service_type_options": services,
        "session_count_options": sorted({s["sessions"] for s in scenarios}),
        "minute_options": _unique([s["minutes"] for s in scenarios]),
        "time_options": time_options,
        "scenarios": scenarios,
        "source_doc_ids": _sources(claims),
        "evidence": _evidence(claims),
        "decisions": decisions,
        "warnings": warnings,
        "identity_unresolved": uncertain_identity,
        "claims": claims,
    }


def _quantity(values, lower=None, upper=None):
    known = sorted(set(values)) if values is not None else []
    if values is not None:
        lower, upper = min(known), max(known)
    exact = upper is not None and lower == upper
    return {"value": lower if exact else None, "min": lower, "max": upper, "alternatives": known if values is not None else None}


def _state_dict(state):
    sessions, dates, undated, low, high = state
    return {"sessions": sessions, "therapy_days_min": max(len(dates), int(undated > 0)), "therapy_days_max": len(dates) + undated, "minutes_min": low, "minutes_max": high}


def _aggregate(events):
    states = {(0, frozenset(), 0, 0.0, 0.0)}
    exact = True
    for event in events:
        next_states = set()
        for state, scenario in product(states, event["scenarios"]):
            count, dates, undated, low, high = state
            amount = scenario["minutes"]
            included = scenario["sessions"]
            day = event.get("date")
            next_states.add((count + included, dates | ({day} if included and day else set()), undated + int(included and not day), low + (amount or 0), high + amount if high is not None and amount is not None else None))
            if len(next_states) > SCENARIO_LIMIT:
                break
        if len(next_states) > SCENARIO_LIMIT:
            exact = False
            break
        states = next_states
    if exact:
        scenarios = _unique([_state_dict(s) for s in states])
        sessions = _quantity([s["sessions"] for s in scenarios])
        days_low = min(s["therapy_days_min"] for s in scenarios)
        days_high = max(s["therapy_days_max"] for s in scenarios)
        day_values = [s["therapy_days_min"] for s in scenarios] if all(s["therapy_days_min"] == s["therapy_days_max"] for s in scenarios) else None
        days = _quantity(day_values, days_low, days_high)
        low = min(s["minutes_min"] for s in scenarios)
        high = None if any(s["minutes_max"] is None for s in scenarios) else max(s["minutes_max"] for s in scenarios)
        values = [s["minutes_min"] for s in scenarios] if all(s["minutes_min"] == s["minutes_max"] for s in scenarios) else None
        minutes = _quantity(values, low, high)
    else:
        sessions = _quantity(None, sum(min(s["sessions"] for s in e["scenarios"]) for e in events), sum(max(s["sessions"] for s in e["scenarios"]) for e in events))
        certain_dates = {e["date"] for e in events if e.get("date") and all(s["sessions"] for s in e["scenarios"])}
        possible_dates = {e["date"] for e in events if e.get("date") and any(s["sessions"] for s in e["scenarios"])}
        undated = sum(not e.get("date") and any(s["sessions"] for s in e["scenarios"]) for e in events)
        days = _quantity(None, len(certain_dates), len(possible_dates) + undated)
        low = sum(min(s["minutes"] or 0 for s in e["scenarios"]) for e in events)
        high = None if any(s["minutes"] is None for e in events for s in e["scenarios"]) else sum(max(s["minutes"] for s in e["scenarios"]) for e in events)
        minutes = _quantity(None, low, high)
        scenarios = [{"sessions": None, "therapy_days_min": days["min"], "therapy_days_max": days["max"], "minutes_min": low, "minutes_max": high}]
    warnings = []
    if not exact:
        warnings.append("Scenario limit reached; reported bounds are conservative and may include unsupported combinations")
    if any(e.get("identity_unresolved") for e in events):
        warnings.append("Unresolved event identity may affect counts; distinct provisional records are currently separate")
        established = [e for e in events if not e.get("identity_unresolved")]
        established_metrics = _aggregate(established)
        sessions = _quantity(None, max(established_metrics["sessions"]["min"], days["min"]), sessions["max"])
        minutes = _quantity(None, established_metrics["minutes"]["min"], minutes["max"])
        exact = False
        scenarios = [{"sessions": None, "therapy_days_min": days["min"], "therapy_days_max": days["max"], "minutes_min": minutes["min"], "minutes_max": minutes["max"]}]
    overlaps = []
    by_day = defaultdict(list)
    for event in events:
        if event.get("date") and any(s["sessions"] for s in event["scenarios"]):
            by_day[event["date"]].append(event)
    for day_events in by_day.values():
        for index, left in enumerate(day_events):
            for right in day_events[index + 1:]:
                if any(_intersection(a, b) for a, b in product(left.get("time_options", []), right.get("time_options", []))):
                    overlaps.append([left["event_id"], right["event_id"]])
    if overlaps:
        warnings.append("Distinct encounter accounts overlap in patient therapy time; total minutes need reconciliation")
        minutes = _quantity(None, 0, minutes["max"])
        exact = False
        scenarios = [{"sessions": None, "therapy_days_min": days["min"], "therapy_days_max": days["max"], "minutes_min": minutes["min"], "minutes_max": minutes["max"]}]
    hours = {k: (v / 60 if v is not None else None) for k, v in minutes.items() if k != "alternatives"}
    hours["alternatives"] = [v / 60 for v in minutes["alternatives"]] if minutes["alternatives"] is not None else None
    return {"sessions": sessions, "therapy_days": days, "minutes": minutes, "hours": hours, "scenarios": scenarios, "exact_scenarios": exact, "overlapping_event_pairs": overlaps, "event_ids": [e["event_id"] for e in events], "source_doc_ids": sorted({d for e in events for d in e["source_doc_ids"]}), "warnings": warnings}


def _plan_for_period(plans, start, end):
    applicable = []
    starts = sorted({p["effective_start"] for p in plans if p.get("effective_start")})
    for plan in plans:
        begin = plan.get("effective_start")
        if not begin or begin > end:
            continue
        finish = plan.get("effective_end")
        if finish is None:
            subsequent = [s for s in starts if s > begin]
            finish = (date.fromisoformat(subsequent[0]) - timedelta(days=1)).isoformat() if subsequent else None
        if finish is not None and finish < start:
            continue
        applicable.append((plan, begin <= start and (finish is None or finish >= end)))
    signatures = _unique([{k: p.get(k) for k in ("min_days", "min_minutes", "week_start", "eligible_services", "patient_present_required")} for p, _ in applicable])
    if not applicable:
        return None, "No effective treatment-plan requirement covers this period"
    if len(signatures) != 1 or not any(covers for _, covers in applicable):
        return None, "Treatment-plan changes or conflicting applicability within the week require an explicit interpretation; no proration assumed"
    return applicable[0][0], None


def _adherence(metrics, plan, reason=None):
    if plan is None:
        return {"status": "cannot_determine", "reason": reason, "plan_source_doc_ids": []}
    if plan.get("week_start", "monday").lower() != "monday" or plan.get("min_days") is None or plan.get("min_minutes") is None:
        return {"status": "cannot_determine", "reason": "Plan thresholds or week convention are missing or unsupported", "plan_source_doc_ids": [plan.get("doc_id")]}
    if set(plan.get("eligible_services", [])) != THERAPY_SERVICES or not plan.get("patient_present_required", True):
        return {"status": "cannot_determine", "reason": "This first calculation supports patient-present individual, group, and family therapy; this plan has a different eligibility rule", "plan_source_doc_ids": [plan.get("doc_id")]}
    target_days, target_minutes = plan["min_days"], plan["min_minutes"]
    outcomes = set()
    for scenario in metrics["scenarios"]:
        if scenario["therapy_days_min"] >= target_days and scenario["minutes_min"] >= target_minutes:
            outcomes.add(True)
        elif scenario["therapy_days_max"] < target_days or (scenario["minutes_max"] is not None and scenario["minutes_max"] < target_minutes):
            outcomes.add(False)
        else:
            outcomes.update((False, True))
    status = "met" if outcomes == {True} else "not_met" if outcomes == {False} else "cannot_determine"
    return {"status": status, "min_days": target_days, "min_minutes": target_minutes, "plan_source_doc_ids": [plan.get("doc_id")], "reason": "Evaluated the conjunction of day and minute requirements across retained scenarios"}


def calculate_metrics(events, plans, start_date=None, end_date=None):
    """Calculate totals and Monday–Sunday weeks over an inclusive review window."""
    episode_starts = [p["episode_start"] for p in plans if p.get("episode_start")]
    episode_ends = [p["episode_end"] for p in plans if p.get("episode_end")]
    dated = [e["date"] for e in events if e.get("date")]
    start_date = _iso(start_date) or min(episode_starts or dated, default=None)
    end_date = _iso(end_date) or max(episode_ends or dated, default=None)
    if start_date and end_date and start_date > end_date:
        raise ValueError("Review start date must not follow end date")
    selected = [e for e in events if not e.get("date") or ((not start_date or e["date"] >= start_date) and (not end_date or e["date"] <= end_date))]
    if start_date or end_date:
        # An undated event might fall outside a bounded question's review period.
        selected = [{**e, "scenarios": _unique(e["scenarios"] + [{**s, "sessions": 0, "minutes": 0} for s in e["scenarios"]])} if not e.get("date") else e for e in selected]
    by_date = {day: _aggregate([e for e in selected if e.get("date") == day]) for day in sorted({e["date"] for e in selected if e.get("date")})}
    by_service = {}
    for service in sorted({s for e in selected for s in e.get("service_type_options", [e["service_type"]])}):
        scoped = []
        for event in selected:
            item = dict(event)
            item["scenarios"] = [dict(s) if s["service_type"] == service else {**s, "sessions": 0, "minutes": 0} for s in event["scenarios"]]
            if any(s["sessions"] for s in item["scenarios"]):
                scoped.append(item)
        by_service[service] = _aggregate(scoped)
    weeks = []
    if start_date and end_date:
        start = date.fromisoformat(start_date)
        monday = start - timedelta(days=start.weekday())
        last = date.fromisoformat(end_date)
        while monday <= last:
            sunday = monday + timedelta(days=6)
            period_start, period_end = max(start_date, monday.isoformat()), min(end_date, sunday.isoformat())
            week_events = [e for e in selected if e.get("date") and period_start <= e["date"] <= period_end]
            metrics = _aggregate(week_events)
            plan, reason = _plan_for_period(plans, period_start, period_end)
            adherence = _adherence(metrics, plan, reason)
            if any(not e.get("date") and any(s["sessions"] for s in e["scenarios"]) for e in selected) and adherence["status"] == "not_met":
                adherence.update(status="cannot_determine", reason="Undated eligible events may belong to this week")
            weeks.append({"week_start": monday.isoformat(), "week_end": sunday.isoformat(), "review_start": period_start, "review_end": period_end, "partial_review_week": period_start != monday.isoformat() or period_end != sunday.isoformat(), **metrics, "plan_adherence": adherence})
            monday += timedelta(days=7)
    return {"review_start": start_date, "review_end": end_date, "overall": _aggregate(selected), "by_service_type": by_service, "by_date": by_date, "weeks": weeks, "undated_event_ids": [e["event_id"] for e in selected if not e.get("date")]}


def _assessments(items):
    groups = defaultdict(list)
    for index, item in enumerate(items):
        key = (item.get("source_form_id"), item.get("instrument")) if item.get("source_form_id") else ("unidentified", item.get("doc_id"), index)
        groups[key].append(item)
    results = []
    for key, claims in sorted(groups.items(), key=lambda pair: str(pair[0])):
        values = _unique([c.get("score") for c in claims if c.get("score") is not None]) or [None]
        dates = sorted({c["completed_date"] for c in claims if c.get("completed_date")})
        first = claims[0]
        issues = []
        if len(values) > 1:
            issues.append("Conflicting scores for one source assessment")
        if len(dates) > 1:
            issues.append("Conflicting completion dates for one source assessment")
        results.append({"assessment_id": "assessment-" + sha256(json.dumps(key).encode()).hexdigest()[:16], "source_form_id": first.get("source_form_id"), "instrument": first.get("instrument"), "completed_date": dates[0] if len(dates) == 1 else None, "completed_date_options": dates, "score": values[0] if len(values) == 1 else None, "score_alternatives": values, "interpretations": sorted({c.get("interpretation", "") for c in claims}), "source_doc_ids": _sources(claims), "evidence": _evidence(claims), "warnings": issues})
    return results


def build_abstraction(patient, records):
    """Reconcile already-extracted records for one resolved patient."""
    encounters, plans, assessments, observations = [], [], [], []
    warnings = []
    for record in records:
        encounters.extend(record.get("encounters", []))
        plans.extend(record.get("plans", []))
        assessments.extend(record.get("assessments", []))
        observations.extend(record.get("observations", []))
        warnings.extend(record.get("warnings", []))
    appointment_to_encounter = defaultdict(set)
    for claim in encounters:
        if claim.get("appointment_id") and claim.get("encounter_id"):
            appointment_to_encounter[claim["appointment_id"]].add(claim["encounter_id"])
    grouped = defaultdict(list)
    uncertain_keys = set()
    for index, claim in enumerate(encounters):
        encounter = claim.get("encounter_id")
        appointment = claim.get("appointment_id")
        if not encounter and appointment and len(appointment_to_encounter[appointment]) == 1:
            encounter = next(iter(appointment_to_encounter[appointment]))
        if encounter:
            key = f"encounter:{encounter}"
        elif appointment:
            key = f"appointment:{appointment}"
        else:
            payload = {k: v for k, v in claim.items() if k not in {"doc_id", "source_doc_id", "source_path"}}
            key = "provisional:" + sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:20]
            uncertain_keys.add(key)
        grouped[key].append(claim)
    events = [_reconcile_event(key, grouped[key], key in uncertain_keys) for key in sorted(grouped)]
    events.sort(key=lambda event: (event["date"] or "9999", event["event_id"]))
    plans = _unique(plans)
    observations = _unique(observations)
    abstraction = {"patient": patient, "events": events, "plans": plans, "assessments": _assessments(assessments), "observations": observations, "warnings": sorted(set(warnings)), "rule_version": "clinical-v1"}
    abstraction["metrics"] = calculate_metrics(events, plans)
    return abstraction
