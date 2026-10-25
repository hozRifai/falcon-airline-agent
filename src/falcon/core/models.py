"""Shared data models (pydantic). The MCP servers return these as plain JSON."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field

FareCode = Literal["SPECIAL", "SAVER", "FLEX"]
BookingStatus = Literal["confirmed", "cancelled", "completed"]
FlightState = Literal["scheduled", "on_time", "delayed", "boarding", "departed", "landed", "cancelled"]


class FareRules(BaseModel):
    code: FareCode
    name_en: str
    name_ar: str
    bag_kg: int
    free_changes: int
    max_changes: int
    change_fee: Decimal
    refund_credit_allowed: bool
    refund_credit_fee: Decimal
    refund_cash_allowed: bool
    refund_cash_fee: Decimal
    deadline_hours: int
    price_multiplier: Decimal


class Passenger(BaseModel):
    id: int
    first_name: str
    last_name: str
    pax_type: Literal["adult", "child", "infant"] = "adult"
    special_request: str | None = None
    cancelled_at: datetime | None = None  # cancelled on a booking that stays confirmed for the others (decision #60)


class Segment(BaseModel):
    seq: int
    flight_instance_id: str
    flight_number: str
    origin: str
    destination: str
    dep_at: datetime
    arr_at: datetime
    fare_per_pax: Decimal


class Extra(BaseModel):
    id: int
    passenger_id: int
    kind: str
    kg: int
    price: Decimal


class Booking(BaseModel):
    pnr: str
    customer_id: str
    status: BookingStatus
    fare_family: FareCode
    currency: str = "AED"
    total_paid: Decimal
    changes_used: int = 0
    created_at: datetime
    passengers: list[Passenger] = Field(default_factory=list)
    segments: list[Segment] = Field(default_factory=list)
    extras: list[Extra] = Field(default_factory=list)
    cancelled_at: datetime | None = None
    refund_method: str | None = None
    refund_amount: Decimal | None = None


class FlightStatus(BaseModel):
    flight_instance_id: str
    status: FlightState
    delay_minutes: int = 0
    reason: str | None = None
    updated_at: datetime | None = None


class Quote(BaseModel):
    kind: Literal["cancel", "change", "bag", "book"]
    pnr: str
    allowed: bool
    reason_code: str
    reason_en: str
    reason_ar: str
    currency: str = "AED"
    fee: Decimal = Decimal("0")
    refund: Decimal = Decimal("0")
    amount_due: Decimal = Decimal("0")
    refund_method: str | None = None
    hours_to_departure: float | None = None
    deadline_at: datetime | None = None
    waiver_possible: bool = False
    details: dict = Field(default_factory=dict)
