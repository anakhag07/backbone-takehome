"""Typed extraction contracts shared by the model boundary and saved cache."""

from datetime import date
import re
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator


def _iso_date(value: str) -> str:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("Dates must use YYYY-MM-DD.")
    date.fromisoformat(value)
    return value


def _clock_time(value: str) -> str:
    if not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", value):
        raise ValueError("Times must use HH:MM on a 24-hour clock.")
    return value


IsoDate = Annotated[str, AfterValidator(_iso_date)]
ClockTime = Annotated[str, AfterValidator(_clock_time)]
Nonnegative = Annotated[float, Field(ge=0)]
ServiceType = Literal[
    "individual", "group", "family", "medication", "care_coordination",
    "collateral", "other", "unknown",
]


class Contract(BaseModel):
    # Structured Outputs requires every object field to be required and extra keys
    # forbidden. Nullable fields below deliberately have no Python defaults. Numeric
    # minimum/maximum and array minItems are supported by current non-fine-tuned models.
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Interval(Contract):
    start: ClockTime
    end: ClockTime

    @model_validator(mode="after")
    def ordered(self):
        if self.end <= self.start:
            raise ValueError("Intervals must have positive duration within one service day.")
        return self


class PatientMention(Contract):
    ref: str
    name: str | None
    dob: IsoDate | None
    mrn: str | None
    issuer: str | None
    evidence: list[str] = Field(min_length=1)


class EncounterClaim(Contract):
    patient_ref: str
    encounter_id: str | None
    appointment_id: str | None
    service_date: IsoDate | None
    service_type: ServiceType
    record_kind: Literal[
        "clinical_note", "attendance", "correction", "copy", "schedule", "draft",
        "charge", "other",
    ]
    record_status: Literal["final", "draft", "unknown"]
    attendance: Literal["attended", "absent", "cancelled", "scheduled", "unknown"]
    patient_present: bool | None
    contact_intervals: list[Interval]
    presence_intervals: list[Interval]
    therapy_intervals: list[Interval]
    excluded_intervals: list[Interval]
    reported_minutes: Nonnegative | None
    corrects_fields: list[str]
    corrects_document_id: str | None
    is_copy: bool
    evidence: list[str] = Field(min_length=1)
    narrative: str
    uncertainty: str | None


class TreatmentPlan(Contract):
    patient_ref: str
    effective_start: IsoDate | None
    effective_end: IsoDate | None
    episode_start: IsoDate | None
    episode_end: IsoDate | None
    min_days: Annotated[int, Field(ge=0, le=7)] | None
    min_minutes: Nonnegative | None
    week_start: Literal[
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        "unknown",
    ]
    eligible_services: list[ServiceType]
    patient_present_required: bool
    evidence: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def ordered_dates(self):
        for first, last in (
            (self.effective_start, self.effective_end),
            (self.episode_start, self.episode_end),
        ):
            if first is not None and last is not None and last < first:
                raise ValueError("Plan end dates cannot precede start dates.")
        return self


class Assessment(Contract):
    patient_ref: str
    source_form_id: str | None
    instrument: str
    completed_date: IsoDate | None
    score: float | None
    interpretation: str
    evidence: list[str] = Field(min_length=1)


class Observation(Contract):
    patient_ref: str
    encounter_id: str | None
    date: IsoDate | None
    text: str
    evidence: list[str] = Field(min_length=1)


class DocumentExtraction(Contract):
    patients: list[PatientMention]
    encounters: list[EncounterClaim]
    plans: list[TreatmentPlan]
    assessments: list[Assessment]
    observations: list[Observation]
    warnings: list[str]

    @model_validator(mode="after")
    def linked_patients(self):
        refs = [patient.ref for patient in self.patients]
        if any(not ref.strip() for ref in refs) or len(refs) != len(set(refs)):
            raise ValueError("Patient references must be nonempty and unique within the document.")
        for collection in (self.encounters, self.plans, self.assessments, self.observations):
            for item in collection:
                if item.patient_ref not in refs:
                    raise ValueError(f"Unknown patient_ref: {item.patient_ref}")
        return self


class QuestionAnswer(Contract):
    answer: str
    status: Literal["answered", "partial", "unresolved"]
    source_doc_ids: list[str]
    calculation_keys: list[str]
    limitations: list[str]


class QuestionScope(Contract):
    question_id: str
    patient_ids: list[str]
    start_date: IsoDate | None
    end_date: IsoDate | None
    focus_dates: list[IsoDate]
    limitations: list[str]

    @model_validator(mode="after")
    def valid_scope(self):
        if self.start_date and self.end_date and self.end_date < self.start_date:
            raise ValueError("Question end date cannot precede its start date.")
        for value in self.focus_dates:
            if self.start_date and value < self.start_date:
                raise ValueError("Focus date falls before the question's start date.")
            if self.end_date and value > self.end_date:
                raise ValueError("Focus date falls after the question's end date.")
        if len(self.patient_ids) != len(set(self.patient_ids)):
            raise ValueError("Question patient IDs cannot be duplicated.")
        if not self.patient_ids and not self.limitations:
            raise ValueError("An unavailable patient scope requires an explicit limitation.")
        return self


class QuestionPlan(Contract):
    scopes: list[QuestionScope]
