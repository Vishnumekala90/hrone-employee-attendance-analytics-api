"""
Employee Attendance & Analytics API  (FastAPI + MongoDB)

Run:  uvicorn app.main:app --port 8000
Env:  MONGO_URI, MONGO_DB  (a local .env is loaded for convenience; real environment variables win)

Layout of this single file
  1. configuration / database handle
  2. time + rule helpers (R1 - R5)         <- pure functions, no database
  3. request models and validation
  4. serialisation (BSON -> API shapes)
  5. indexes (created at startup, idempotent)
  6. employees / attendance endpoints
  7. analytics pipelines + endpoints       <- builders are shared with /admin/explain
  8. explain endpoint
"""
import calendar
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from datetime import date as date_cls
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Literal, Optional

from bson import json_util
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Path, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, StrictInt, field_validator, model_validator
from pymongo import ASCENDING, DESCENDING, MongoClient, ReturnDocument
from pymongo.errors import DuplicateKeyError, PyMongoError

# --------------------------------------------------------------------------- #
# 1. Configuration
# --------------------------------------------------------------------------- #
load_dotenv()  # override=False: real environment variables always win

log = logging.getLogger("attendance")

# tz_aware=True: PyMongo hands back aware (UTC) datetimes instead of naive ones, so we never
# have to guess which zone a stored instant is in.
client = MongoClient(
    os.getenv("MONGO_URI", "mongodb://localhost:27017"),
    tz_aware=True,
    tzinfo=timezone.utc,
    serverSelectionTimeoutMS=5000,
)
db = client[os.getenv("MONGO_DB", "attendance_db")]

IST = timezone(timedelta(hours=5, minutes=30))
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)

MIN_MS = 100_000_000_000          # anything below looks like epoch *seconds* -> rejected
MAX_MS = 4_102_444_800_000        # 2100-01-01
GRACE_SECONDS = 10 * 60           # R2
MIN_OVERTIME_MINUTES = 30         # R3
HALF_DAY_HUNDREDTHS = 450         # R5: rounded work_hours < 4.50
MAX_SHIFT_SECONDS = 24 * 3600     # punch-out must be within 24h of punch-in
PRESENCE = ["PRESENT", "WFH", "ON_DUTY"]  # R6
WEEKDAYS_MONGO = [2, 3, 4, 5, 6]  # $dayOfWeek: 1 = Sunday ... 7 = Saturday  -> Mon..Fri

DATE_RE = r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$"
MONTH_RE = r"^[0-9]{4}-(0[1-9]|1[0-2])$"
HHMM_RE = r"^([01][0-9]|2[0-3]):[0-5][0-9]$"

StatusT = Literal["PRESENT", "ABSENT", "LEAVE", "WFH", "ON_DUTY"]
PresenceStatusT = Literal["PRESENT", "WFH", "ON_DUTY"]
EpochMs = Annotated[StrictInt, Field(ge=MIN_MS, le=MAX_MS)]  # StrictInt: floats/strings/bools -> 422


# --------------------------------------------------------------------------- #
# 2. Time and rule helpers (pure)
# --------------------------------------------------------------------------- #
def ms_to_dt(ms: int) -> datetime:
    """Epoch ms -> aware UTC datetime, truncated to whole seconds (R1)."""
    return EPOCH + timedelta(seconds=ms // 1000)


def dt_to_ms(dt: Optional[datetime]) -> Optional[int]:
    """Datetime (aware or naive-UTC) -> integer epoch milliseconds, exactly (no float maths)."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (dt - EPOCH) // timedelta(milliseconds=1)


def now_dt() -> datetime:
    """Current time as aware UTC, whole milliseconds (BSON precision)."""
    n = datetime.now(timezone.utc)
    return n.replace(microsecond=(n.microsecond // 1000) * 1000)


def hhmm_to_seconds(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 3600 + int(m) * 60


def is_overnight(shift_start: str, shift_end: str) -> bool:
    return hhmm_to_seconds(shift_end) <= hhmm_to_seconds(shift_start)


def attendance_date(punch_in: datetime, shift_start: str, shift_end: str) -> str:
    """R1: IST calendar date of the punch-in; for overnight shifts a punch-in earlier than
    shift_end (IST time of day) belongs to the shift that started the previous day."""
    local = punch_in.astimezone(IST)
    if is_overnight(shift_start, shift_end):
        tod = local.hour * 3600 + local.minute * 60 + local.second
        if tod < hhmm_to_seconds(shift_end):
            local -= timedelta(days=1)
    return local.date().isoformat()


def _at_ist(date_str: str, hhmm: str) -> datetime:
    y, mo, d = map(int, date_str.split("-"))
    h, m = map(int, hhmm.split(":"))
    return datetime(y, mo, d, h, m, tzinfo=IST)


def shift_start_dt(date_str: str, shift_start: str) -> datetime:
    return _at_ist(date_str, shift_start)


def shift_end_dt(date_str: str, shift_start: str, shift_end: str) -> datetime:
    end = _at_ist(date_str, shift_end)
    if is_overnight(shift_start, shift_end):
        end += timedelta(days=1)  # an overnight shift ends on the next calendar day
    return end


def compute_late_minutes(punch_in: datetime, date_str: str, shift_start: str) -> int:
    """R2: late only if strictly more than 10:00 after shift start; then whole minutes since
    the shift start (not since the end of the grace period)."""
    secs = (punch_in - shift_start_dt(date_str, shift_start)) // timedelta(seconds=1)
    return secs // 60 if secs > GRACE_SECONDS else 0


def compute_work_hours(punch_in: datetime, punch_out: datetime) -> float:
    """R4: seconds / 3600 rounded to 2 decimals HALF-UP, using integer arithmetic only
    (Python's round() is banker's rounding and float division can land on x.xx4999...)."""
    secs = (punch_out - punch_in) // timedelta(seconds=1)
    hundredths = (secs * 200 + 3600) // 7200  # floor(secs*100/3600 + 1/2)
    return hundredths / 100


def compute_overtime(punch_out: datetime, date_str: str, shift_start: str, shift_end: str) -> int:
    """R3: whole minutes after shift end, counted only when >= 30."""
    secs = (punch_out - shift_end_dt(date_str, shift_start, shift_end)) // timedelta(seconds=1)
    minutes = secs // 60
    return minutes if minutes >= MIN_OVERTIME_MINUTES else 0


def compute_half_day(work_hours: float) -> bool:
    """R5: judged on the ROUNDED work hours."""
    return round(work_hours * 100) < HALF_DAY_HUNDREDTHS


def derived_fields(
    status: str,
    punch_in: Optional[datetime],
    punch_out: Optional[datetime],
    date_str: str,
    shift_start: str,
    shift_end: str,
) -> dict:
    """All four stored derived values for a record, from its final state (used by punch-in,
    punch-out and regularize so the three can never disagree)."""
    if status not in PRESENCE or punch_in is None:
        return {"work_hours": None, "late_minutes": 0, "overtime_minutes": 0, "half_day": False}
    out = {
        "work_hours": None,
        "late_minutes": compute_late_minutes(punch_in, date_str, shift_start),
        "overtime_minutes": 0,
        "half_day": False,
    }
    if punch_out is not None:
        wh = compute_work_hours(punch_in, punch_out)
        out["work_hours"] = wh
        out["overtime_minutes"] = compute_overtime(punch_out, date_str, shift_start, shift_end)
        out["half_day"] = compute_half_day(wh)
    return out


def parse_date_str(value: str) -> date_cls:
    """Strict YYYY-MM-DD -> date (raises ValueError)."""
    if not re.fullmatch(DATE_RE, value):
        raise ValueError("date must be YYYY-MM-DD")
    return date_cls(int(value[0:4]), int(value[5:7]), int(value[8:10]))


def month_bounds(month: str) -> tuple[str, str]:
    """'2026-07' -> ('2026-07-01', '2026-07-31')."""
    y, m = int(month[:4]), int(month[5:7])
    if y < 1:
        raise ValueError("year out of range")
    last = calendar.monthrange(y, m)[1]
    return f"{y:04d}-{m:02d}-01", f"{y:04d}-{m:02d}-{last:02d}"


def count_weekdays(start: date_cls, end: date_cls) -> int:
    """Mon-Fri days in [start, end] inclusive (R7), without looping over the days."""
    if end < start:
        return 0
    days = (end - start).days + 1
    full, rest = divmod(days, 7)
    count = full * 5
    wd = start.weekday()  # Monday = 0
    for i in range(rest):  # at most 6 iterations
        if (wd + i) % 7 < 5:
            count += 1
    return count


# --------------------------------------------------------------------------- #
# 3. Validation helpers and request models
# --------------------------------------------------------------------------- #
def invalid(loc: tuple, msg: str) -> RequestValidationError:
    """A 422 with FastAPI's default list-shaped body."""
    return RequestValidationError([{"type": "value_error", "loc": loc, "msg": msg, "input": None}])


def checked_date(value: str, loc: tuple) -> str:
    try:
        parse_date_str(value)
    except ValueError:
        raise invalid(loc, "must be a valid date in YYYY-MM-DD format")
    return value


def checked_month(value: str) -> tuple[str, str]:
    try:
        return month_bounds(value)
    except ValueError:
        raise invalid(("query", "month"), "must be a valid month in YYYY-MM format")


class EmployeeCreate(BaseModel):
    emp_code: str = Field(pattern=r"^EMP[0-9]{4,6}$")
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(max_length=120, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    department: str = Field(min_length=1, max_length=50)
    shift_start: str = Field("09:30", pattern=HHMM_RE)
    shift_end: str = Field("18:30", pattern=HHMM_RE)
    joined_on: str

    @field_validator("joined_on")
    @classmethod
    def _joined_on_is_a_date(cls, v: str) -> str:
        parse_date_str(v)
        return v

    @model_validator(mode="after")
    def _shift_times_differ(self):
        if self.shift_start == self.shift_end:
            raise ValueError("shift_start must differ from shift_end")
        return self


class PunchInRequest(BaseModel):
    emp_code: str
    punched_at: Optional[EpochMs] = None
    status: PresenceStatusT = "PRESENT"


class PunchOutRequest(BaseModel):
    emp_code: str
    punched_at: Optional[EpochMs] = None


class RegularizeRequest(BaseModel):
    # Derived fields are not declared, so anything a client sends for them is ignored.
    status: Optional[StatusT] = None
    punch_in: Optional[EpochMs] = None
    punch_out: Optional[EpochMs] = None
    reason: str = Field(min_length=5, max_length=200)
    regularized_by: str = Field(min_length=1, max_length=50)


# --------------------------------------------------------------------------- #
# 4. Serialisation: BSON documents -> API shapes
# --------------------------------------------------------------------------- #
def employee_out(d: dict) -> dict:
    return {
        "emp_code": d["emp_code"],
        "name": d.get("name"),
        "email": d.get("email"),
        "department": d.get("department"),
        "shift_start": d.get("shift_start"),
        "shift_end": d.get("shift_end"),
        "joined_on": d.get("joined_on"),
        "created_at": dt_to_ms(d.get("created_at")),
    }


def _history_value(field: str, v: Any) -> Any:
    # punch times are BSON dates in MongoDB but epoch ms in the API, also inside `changes`
    if field in ("punch_in", "punch_out"):
        return dt_to_ms(v) if isinstance(v, datetime) else v
    return v


def attendance_out(d: dict) -> dict:
    """Never exposes _id; fills in legacy gaps (missing history / half_day / counters)."""
    history = []
    for h in d.get("history") or []:
        changes = {
            k: {"from": _history_value(k, c.get("from")), "to": _history_value(k, c.get("to"))}
            for k, c in (h.get("changes") or {}).items()
        }
        history.append({"at": dt_to_ms(h.get("at")), "by": h.get("by"), "reason": h.get("reason"), "changes": changes})
    return {
        "emp_code": d["emp_code"],
        "date": d["date"],
        "status": d["status"],
        "punch_in": dt_to_ms(d.get("punch_in")),
        "punch_out": dt_to_ms(d.get("punch_out")),
        "work_hours": d.get("work_hours"),
        "late_minutes": d.get("late_minutes") or 0,
        "overtime_minutes": d.get("overtime_minutes") or 0,
        "half_day": bool(d.get("half_day", False)),
        "history": history,
    }


# --------------------------------------------------------------------------- #
# 5. Indexes (idempotent: same keys + options -> no-op)
# --------------------------------------------------------------------------- #
def ensure_indexes() -> None:
    emp, att = db.employees, db.attendance_logs
    # employees
    emp.create_index([("emp_code", ASCENDING)], unique=True, name="uniq_emp_code")
    emp.create_index([("department", ASCENDING), ("emp_code", ASCENDING)], name="dept_emp_code")
    emp.create_index([("department", ASCENDING), ("joined_on", ASCENDING)], name="dept_joined_on")
    emp.create_index([("joined_on", ASCENDING)], name="joined_on")
    # attendance_logs
    att.create_index([("emp_code", ASCENDING), ("date", ASCENDING)], unique=True, name="uniq_emp_date")
    att.create_index([("date", DESCENDING), ("emp_code", ASCENDING)], name="date_desc_emp")
    att.create_index([("status", ASCENDING), ("date", DESCENDING), ("emp_code", ASCENDING)], name="status_date_emp")
    att.create_index([("date", ASCENDING), ("late_minutes", ASCENDING), ("emp_code", ASCENDING)], name="late_cover")


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        ensure_indexes()
    except PyMongoError as exc:  # keep the process up; /health will report 503 until Mongo answers
        log.error("could not create indexes at startup: %s", exc)
    yield


app = FastAPI(title="Employee Attendance & Analytics API", version="2.0.0", lifespan=lifespan)


@app.exception_handler(PyMongoError)
async def mongo_error_handler(_: Request, exc: PyMongoError):
    """A database outage is a 503 with the standard {"detail": ...} body, never a bare 500."""
    log.error("database error: %s", exc)
    return JSONResponse(status_code=503, content={"detail": "database unavailable"})


# --------------------------------------------------------------------------- #
# 6. Health, employees, attendance
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    try:
        client.admin.command("ping")
    except PyMongoError:
        raise HTTPException(503, "database unavailable")
    return {"status": "ok"}


@app.post("/employees", status_code=201)
def create_employee(body: EmployeeCreate):
    doc = body.model_dump()
    doc["created_at"] = now_dt()
    try:
        db.employees.insert_one(doc)  # the unique index is the only thing that decides "duplicate"
    except DuplicateKeyError:
        raise HTTPException(409, "emp_code already exists")
    return employee_out(doc)


@app.get("/employees")
def list_employees(
    department: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    q: dict = {} if department is None else {"department": department}
    total = db.employees.count_documents(q)
    cur = db.employees.find(q, {"_id": 0}).sort("emp_code", ASCENDING).skip((page - 1) * page_size).limit(page_size)
    return {"items": [employee_out(d) for d in cur], "total": total, "page": page, "page_size": page_size}


def get_employee_or_404(emp_code: str) -> dict:
    emp = db.employees.find_one({"emp_code": emp_code}, {"_id": 0})
    if emp is None:
        raise HTTPException(404, "employee not found")
    return emp


@app.post("/attendance/punch-in", status_code=201)
def punch_in(body: PunchInRequest):
    emp = get_employee_or_404(body.emp_code)
    ts = ms_to_dt(body.punched_at) if body.punched_at is not None else ms_to_dt(dt_to_ms(now_dt()))
    date_str = attendance_date(ts, emp["shift_start"], emp["shift_end"])
    doc = {
        "emp_code": body.emp_code,
        "date": date_str,
        "status": body.status,
        "punch_in": ts,
        "punch_out": None,
        **derived_fields(body.status, ts, None, date_str, emp["shift_start"], emp["shift_end"]),
        "history": [],
    }
    try:
        db.attendance_logs.insert_one(doc)  # unique (emp_code, date): exactly one racer wins
    except DuplicateKeyError:
        raise HTTPException(409, "already punched in for this date")
    return attendance_out(doc)


@app.post("/attendance/punch-out")
def punch_out(body: PunchOutRequest):
    emp = get_employee_or_404(body.emp_code)
    ts = ms_to_dt(body.punched_at) if body.punched_at is not None else ms_to_dt(dt_to_ms(now_dt()))
    # Most recent record with punch_in <= ts. A record's date is never later than the IST date
    # of its punch_in, so bounding `date` lets the (emp_code, date) index do the work.
    latest_date = ts.astimezone(IST).date().isoformat()
    rec = db.attendance_logs.find_one(
        {"emp_code": body.emp_code, "date": {"$lte": latest_date}, "punch_in": {"$lte": ts}},
        sort=[("date", DESCENDING), ("punch_in", DESCENDING)],
    )
    if rec is None:
        raise HTTPException(404, "no punch-in found for this employee")
    if rec.get("punch_out") is not None:
        raise HTTPException(409, "already punched out")
    secs = (ts - rec["punch_in"]) // timedelta(seconds=1)
    if secs <= 0:
        raise HTTPException(422, "punched_at must be after punch_in")
    if secs > MAX_SHIFT_SECONDS:
        raise HTTPException(422, "punch-out is more than 24 hours after punch-in")
    fields = derived_fields(rec["status"], rec["punch_in"], ts, rec["date"], emp["shift_start"], emp["shift_end"])
    fields["punch_out"] = ts
    # Compare-and-set: only the request that still sees punch_out == null can win.
    updated = db.attendance_logs.find_one_and_update(
        {"_id": rec["_id"], "punch_out": None},
        {"$set": fields},
        return_document=ReturnDocument.AFTER,
    )
    if updated is None:
        raise HTTPException(409, "already punched out")
    return attendance_out(updated)


def build_attendance_query(
    emp_code: Optional[str],
    date_from: Optional[str],
    date_to: Optional[str],
    status: Optional[str],
    page: int,
    page_size: int,
) -> dict:
    """Shared by GET /attendance and /admin/explain/attendance_list."""
    if date_from is not None:
        checked_date(date_from, ("query", "date_from"))
    if date_to is not None:
        checked_date(date_to, ("query", "date_to"))
    if date_from is not None and date_to is not None and date_from > date_to:
        raise invalid(("query", "date_from"), "date_from must not be after date_to")
    q: dict = {}
    if emp_code is not None:
        q["emp_code"] = emp_code
    if date_from is not None or date_to is not None:
        q["date"] = {}
        if date_from is not None:
            q["date"]["$gte"] = date_from
        if date_to is not None:
            q["date"]["$lte"] = date_to
    if status is not None:
        q["status"] = status
    return {
        "filter": q,
        "sort": [("date", DESCENDING), ("emp_code", ASCENDING)],
        "skip": (page - 1) * page_size,
        "limit": page_size,
    }


@app.get("/attendance")
def list_attendance(
    emp_code: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[StatusT] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    spec = build_attendance_query(emp_code, date_from, date_to, status, page, page_size)
    q = spec["filter"]
    total = db.attendance_logs.count_documents(q) if q else db.attendance_logs.estimated_document_count()
    cur = (
        db.attendance_logs.find(q, {"_id": 0})
        .sort(spec["sort"])
        .skip(spec["skip"])
        .limit(spec["limit"])
    )
    return {"items": [attendance_out(d) for d in cur], "total": total, "page": page, "page_size": page_size}


@app.patch("/attendance/{emp_code}/{date}")
def regularize(emp_code: str, date: Annotated[str, Path(pattern=DATE_RE)], body: RegularizeRequest):
    checked_date(date, ("path", "date"))
    emp = get_employee_or_404(emp_code)
    rec = db.attendance_logs.find_one({"emp_code": emp_code, "date": date})
    if rec is None:
        raise HTTPException(404, "no attendance record for that date")
    s_start, s_end = emp["shift_start"], emp["shift_end"]

    old = {
        "status": rec["status"],
        "punch_in": rec.get("punch_in"),
        "punch_out": rec.get("punch_out"),
        "work_hours": rec.get("work_hours"),
        "late_minutes": rec.get("late_minutes") or 0,
        "overtime_minutes": rec.get("overtime_minutes") or 0,
        "half_day": bool(rec.get("half_day", False)),
    }

    new_status = body.status or old["status"]
    if new_status in PRESENCE:
        new_in = ms_to_dt(body.punch_in) if body.punch_in is not None else old["punch_in"]
        new_out = ms_to_dt(body.punch_out) if body.punch_out is not None else old["punch_out"]
        if new_in is None:
            raise HTTPException(422, "a presence status requires a punch_in")
        if body.punch_in is not None and attendance_date(new_in, s_start, s_end) != date:
            raise HTTPException(422, "punch_in must stay on the record's attendance date")
        if new_out is not None and (body.punch_in is not None or body.punch_out is not None):
            secs = (new_out - new_in) // timedelta(seconds=1)
            if secs <= 0:
                raise HTTPException(422, "punch_out must be after punch_in")
            if secs > MAX_SHIFT_SECONDS:
                raise HTTPException(422, "punch_out must be within 24 hours of punch_in")
    else:  # ABSENT / LEAVE clear both punch times
        if body.punch_in is not None or body.punch_out is not None:
            raise HTTPException(422, f"punch times cannot be supplied with status {new_status}")
        new_in = new_out = None

    new = {"status": new_status, "punch_in": new_in, "punch_out": new_out}
    new.update(derived_fields(new_status, new_in, new_out, date, s_start, s_end))

    if all(new[k] == old[k] for k in ("status", "punch_in", "punch_out")):
        raise HTTPException(422, "request changes nothing")
    changes = {k: {"from": old[k], "to": new[k]} for k in old if old[k] != new[k]}
    entry = {"at": now_dt(), "by": body.regularized_by, "reason": body.reason, "changes": changes}

    # Optimistic concurrency: the update only applies if the record is still exactly as we read
    # it (same history length, same editable fields). A concurrent edit makes this a no-match -> 409,
    # so two corrections can never overwrite each other's history.
    guard: dict = {"_id": rec["_id"], "status": old["status"], "punch_in": old["punch_in"], "punch_out": old["punch_out"]}
    if isinstance(rec.get("history"), list):
        guard["history"] = {"$size": len(rec["history"])}
    else:
        guard["history"] = {"$exists": False}
    updated = db.attendance_logs.find_one_and_update(
        guard, {"$set": new, "$push": {"history": entry}}, return_document=ReturnDocument.AFTER
    )
    if updated is None:
        raise HTTPException(409, "record was modified concurrently; retry")
    return attendance_out(updated)


# --------------------------------------------------------------------------- #
# 7. Analytics. Every endpoint builds its pipeline in a function that /admin/explain reuses.
#    Rounding is half-up everywhere (MongoDB's $round is banker's rounding), done with
#    integer arithmetic: floor(x + 0.5) on values that are exact in binary.
# --------------------------------------------------------------------------- #
def _present2(weekday_only: bool, prefix: str = "") -> dict:
    """Twice the present-day value of one log: 2 for a full day, 1 for a half day, 0 if not
    present. Doubling keeps everything an exact integer until the final division."""
    cond: Any = {"$in": [f"${prefix}status", PRESENCE]}
    if weekday_only:
        dow = {"$dayOfWeek": {"$dateFromString": {"dateString": f"${prefix}date", "onError": None, "onNull": None}}}
        cond = {"$and": [cond, {"$in": [dow, WEEKDAYS_MONGO]}]}
    half = {"$ifNull": [f"${prefix}half_day", False]}
    return {"$cond": [cond, {"$cond": [half, 1, 2]}, 0]}


def _late_flag(prefix: str = "") -> dict:
    return {"$cond": [{"$gt": [{"$ifNull": [f"${prefix}late_minutes", 0]}, 0]}, 1, 0]}


def _status_flag(status: str, prefix: str = "") -> dict:
    return {"$cond": [{"$eq": [f"${prefix}status", status]}, 1, 0]}


def _has_work_hours(prefix: str = "") -> dict:
    return {
        "$and": [
            {"$in": [f"${prefix}status", PRESENCE]},
            {"$in": [{"$type": f"${prefix}work_hours"}, ["double", "int", "long", "decimal"]]},
        ]
    }


def _floor_half_up(expr: Any) -> dict:
    return {"$floor": {"$add": [expr, 0.5]}}


# ---- employee monthly --------------------------------------------------------
def monthly_pipeline(emp_code: str, start: str, end: str) -> list:
    return [
        {"$match": {"emp_code": emp_code, "date": {"$gte": start, "$lte": end}}},
        {
            "$group": {
                "_id": None,
                "present2": {"$sum": _present2(True)},
                "leave_days": {"$sum": _status_flag("LEAVE")},
                "late_count": {"$sum": _late_flag()},
                "late_minutes": {"$sum": {"$ifNull": ["$late_minutes", 0]}},
                "overtime_minutes": {"$sum": {"$ifNull": ["$overtime_minutes", 0]}},
            }
        },
    ]


@app.get("/analytics/employees/{emp_code}/monthly")
def employee_monthly(emp_code: str, month: Annotated[str, Query(pattern=MONTH_RE)]):
    start, end = checked_month(month)
    emp = get_employee_or_404(emp_code)
    row = next(iter(db.attendance_logs.aggregate(monthly_pipeline(emp_code, start, end))), None) or {}
    # R7: working days run from joined_on when that falls inside the month.
    first = max(parse_date_str(start), parse_date_str(emp["joined_on"]))
    working_days = count_weekdays(first, parse_date_str(end))
    present2 = int(row.get("present2", 0))
    pct = None
    if working_days > 0:
        # present/working*100 to 2 decimals half-up, exact: floor(present2*50/wd + 1/2) / 100
        pct = ((present2 * 10000 + working_days) // (2 * working_days)) / 100
    return {
        "emp_code": emp_code,
        "month": month,
        "working_days": working_days,
        "present_days": present2 / 2,
        "leave_days": int(row.get("leave_days", 0)),
        "late_count": int(row.get("late_count", 0)),
        "total_late_minutes": int(row.get("late_minutes", 0)),
        "total_overtime_minutes": int(row.get("overtime_minutes", 0)),
        "attendance_pct": pct,
    }


# ---- department summary ------------------------------------------------------
def summary_pipeline(start: str, end: str, department: Optional[str]) -> list:
    match: dict = {"joined_on": {"$lte": end}}  # R9
    if department is not None:
        match["department"] = department
    per_employee = [
        {"$match": {"date": {"$gte": start, "$lte": end}}},
        {
            "$group": {
                "_id": None,
                "present2": {"$sum": _present2(True)},
                "late_count": {"$sum": _late_flag()},
                "late_minutes": {"$sum": {"$ifNull": ["$late_minutes", 0]}},
                "leave_count": {"$sum": _status_flag("LEAVE")},
                "on_duty_count": {"$sum": _status_flag("ON_DUTY")},
                # work_hours kept as integer hundredths so the mean can be rounded exactly
                "wh_hundredths": {"$sum": {"$cond": [_has_work_hours(), {"$round": [{"$multiply": ["$work_hours", 100]}, 0]}, 0]}},
                "wh_n": {"$sum": {"$cond": [_has_work_hours(), 1, 0]}},
            }
        },
    ]
    return [
        {"$match": match},
        # Start from EMPLOYEES so people with zero logs still produce a row (headcount),
        # then left-join their month of logs through the (emp_code, date) index.
        {"$lookup": {"from": "attendance_logs", "localField": "emp_code", "foreignField": "emp_code", "pipeline": per_employee, "as": "m"}},
        {"$unwind": {"path": "$m", "preserveNullAndEmptyArrays": True}},
        {
            "$group": {
                "_id": "$department",
                "headcount": {"$sum": 1},
                "present2": {"$sum": {"$ifNull": ["$m.present2", 0]}},
                "late_count": {"$sum": {"$ifNull": ["$m.late_count", 0]}},
                "late_minutes": {"$sum": {"$ifNull": ["$m.late_minutes", 0]}},
                "leave_count": {"$sum": {"$ifNull": ["$m.leave_count", 0]}},
                "on_duty_count": {"$sum": {"$ifNull": ["$m.on_duty_count", 0]}},
                "wh_hundredths": {"$sum": {"$ifNull": ["$m.wh_hundredths", 0]}},
                "wh_n": {"$sum": {"$ifNull": ["$m.wh_n", 0]}},
            }
        },
        {"$sort": {"_id": 1}},
        {
            "$project": {
                "_id": 0,
                "department": "$_id",
                "headcount": 1,
                "present_days": {"$divide": ["$present2", 2]},
                "avg_work_hours": {
                    "$cond": [
                        {"$gt": ["$wh_n", 0]},
                        {"$divide": [_floor_half_up({"$divide": ["$wh_hundredths", "$wh_n"]}), 100]},
                        None,
                    ]
                },
                "late_count": 1,
                "total_late_minutes": "$late_minutes",
                "leave_count": 1,
                "on_duty_count": 1,
            }
        },
    ]


@app.get("/analytics/departments/summary")
def department_summary(month: Annotated[str, Query(pattern=MONTH_RE)], department: Optional[str] = None):
    start, end = checked_month(month)
    items = list(db.employees.aggregate(summary_pipeline(start, end, department)))
    return {"month": month, "items": items}


# ---- late leaderboard --------------------------------------------------------
def leaderboard_pipeline(start: str, end: str, limit: int, department: Optional[str]) -> list:
    pipeline: list = [
        {"$match": {"date": {"$gte": start, "$lte": end}, "late_minutes": {"$gt": 0}}},
        {"$group": {"_id": "$emp_code", "total_late_minutes": {"$sum": "$late_minutes"}, "late_count": {"$sum": 1}}},
        {"$lookup": {"from": "employees", "localField": "_id", "foreignField": "emp_code", "as": "e"}},
        {"$unwind": "$e"},  # logs whose employee does not exist are dropped here
    ]
    if department is not None:
        pipeline.append({"$match": {"e.department": department}})  # rank inside the department only
    pipeline += [
        # Standard competition ranking (1, 2, 2, 4): $rank gives ties the same rank and skips.
        {"$setWindowFields": {"sortBy": {"total_late_minutes": -1}, "output": {"rank": {"$rank": {}}}}},
        {"$match": {"rank": {"$lte": limit}}},  # limit applies AFTER ranking
        {"$sort": {"total_late_minutes": -1, "_id": 1}},
        {
            "$project": {
                "_id": 0,
                "rank": 1,
                "emp_code": "$_id",
                "name": "$e.name",
                "department": "$e.department",
                "total_late_minutes": 1,
                "late_count": 1,
            }
        },
    ]
    return pipeline


@app.get("/analytics/leaderboard/late")
def late_leaderboard(
    month: Annotated[str, Query(pattern=MONTH_RE)],
    limit: int = Query(10, ge=1, le=50),
    department: Optional[str] = None,
):
    start, end = checked_month(month)
    rows = list(db.attendance_logs.aggregate(leaderboard_pipeline(start, end, limit, department)))
    for r in rows:
        r["rank"] = int(r["rank"])
        r["total_late_minutes"] = int(r["total_late_minutes"])
    return {"month": month, "items": rows}


# ---- department trend --------------------------------------------------------
def trend_range(frm: str, to: str) -> int:
    """Validates the range and returns the number of calendar days in it."""
    checked_date(frm, ("query", "from"))
    checked_date(to, ("query", "to"))
    days = (parse_date_str(to) - parse_date_str(frm)).days + 1
    if days < 1:
        raise invalid(("query", "to"), "to must not be before from")
    if days > 92:
        raise invalid(("query", "to"), "range must not exceed 92 days")
    return days


def trend_pipeline(department: str, frm: str, days: int) -> list:
    day = {
        "$dateToString": {
            "format": "%Y-%m-%d",
            "date": {"$dateAdd": {"startDate": {"$dateFromString": {"dateString": frm}}, "unit": "day", "amount": "$$i"}},
        }
    }
    to = (parse_date_str(frm) + timedelta(days=days - 1)).isoformat()
    window = {"documents": [-6, 0]}  # current row + up to 6 preceding rows, never before `from`
    return [
        {"$match": {"department": department}},
        {
            "$lookup": {
                "from": "attendance_logs",
                "localField": "emp_code",
                "foreignField": "emp_code",
                "pipeline": [
                    {"$match": {"date": {"$gte": frm, "$lte": to}}},
                    {"$project": {"_id": 0, "date": 1, "status": 1, "half_day": 1, "late_minutes": 1}},
                ],
                "as": "logs",
            }
        },
        {
            "$facet": {
                # per-date totals over the department's logs (<= 92 groups)
                "daily": [
                    {"$unwind": "$logs"},
                    {"$group": {"_id": "$logs.date", "p2": {"$sum": _present2(False, "logs.")}, "late": {"$sum": _late_flag("logs.")}}},
                ],
                # how many employees joined on each date (<= #distinct joined_on values)
                "joins": [{"$group": {"_id": "$joined_on", "n": {"$sum": 1}}}],
            }
        },
        # Gap filling: one row per calendar day from $range, whether or not any log exists.
        {
            "$project": {
                "rows": {
                    "$map": {
                        "input": {"$range": [0, days]},
                        "as": "i",
                        "in": {
                            "$let": {
                                "vars": {"d": day},
                                "in": {
                                    "$let": {
                                        "vars": {"x": {"$arrayElemAt": [{"$filter": {"input": "$daily", "as": "r", "cond": {"$eq": ["$$r._id", "$$d"]}}}, 0]}},
                                        "in": {
                                            "date": "$$d",
                                            "dow": {"$dayOfWeek": {"$dateFromString": {"dateString": "$$d"}}},
                                            "headcount": {
                                                "$reduce": {
                                                    "input": "$joins",
                                                    "initialValue": 0,
                                                    "in": {"$cond": [{"$lte": ["$$this._id", "$$d"]}, {"$add": ["$$value", "$$this.n"]}, "$$value"]},
                                                }
                                            },
                                            "p2": {"$ifNull": ["$$x.p2", 0]},
                                            "late": {"$ifNull": ["$$x.late", 0]},
                                        },
                                    }
                                },
                            }
                        },
                    }
                }
            }
        },
        {"$unwind": "$rows"},
        {"$replaceRoot": {"newRoot": "$rows"}},
        {"$addFields": {"wd": {"$in": ["$dow", WEEKDAYS_MONGO]}}},
        {
            "$addFields": {
                # rate in 1/10000ths, half-up: (p2/2)/headcount*10000 = p2*5000/headcount
                "rs": {
                    "$cond": [
                        {"$and": ["$wd", {"$gt": ["$headcount", 0]}]},
                        _floor_half_up({"$divide": [{"$multiply": ["$p2", 5000]}, "$headcount"]}),
                        None,
                    ]
                }
            }
        },
        {
            "$setWindowFields": {
                "sortBy": {"date": 1},
                "output": {
                    "s": {"$sum": "$rs", "window": window},  # $sum ignores the nulls
                    "c": {"$sum": {"$cond": [{"$ne": ["$rs", None]}, 1, 0]}, "window": window},
                },
            }
        },
        {"$sort": {"date": 1}},
        {
            "$project": {
                "_id": 0,
                "date": 1,
                "is_working_day": "$wd",
                "headcount": 1,
                "present_count": {"$divide": ["$p2", 2]},
                "late_count": "$late",
                "attendance_rate": {"$cond": [{"$ne": ["$rs", None]}, {"$divide": ["$rs", 10000]}, None]},
                "moving_avg_7d": {
                    "$cond": [{"$gt": ["$c", 0]}, {"$divide": [_floor_half_up({"$divide": ["$s", "$c"]}), 10000]}, None]
                },
            }
        },
    ]


@app.get("/analytics/departments/{department}/trend")
def department_trend(
    department: str,
    frm: Annotated[str, Query(alias="from")],
    to: Annotated[str, Query()],
):
    days = trend_range(frm, to)
    if db.employees.count_documents({"department": department}, limit=1) == 0:
        raise HTTPException(404, "unknown department")
    items = list(db.employees.aggregate(trend_pipeline(department, frm, days)))
    return {"department": department, "items": items}


# --------------------------------------------------------------------------- #
# 8. Explain
# --------------------------------------------------------------------------- #
def _require(value: Any, name: str):
    if value is None:
        raise invalid(("query", name), "field required")
    return value


@app.get("/admin/explain/{endpoint}")
def explain_endpoint(
    endpoint: Literal["attendance_list", "employee_monthly", "department_summary", "late_leaderboard", "department_trend"],
    emp_code: Optional[str] = None,
    month: Annotated[Optional[str], Query(pattern=MONTH_RE)] = None,
    department: Optional[str] = None,
    limit: int = Query(10, ge=1, le=50),
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[StatusT] = None,
    frm: Annotated[Optional[str], Query(alias="from")] = None,
    to: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    if endpoint == "attendance_list":
        spec = build_attendance_query(emp_code, date_from, date_to, status, page, page_size)
        collection = "attendance_logs"
        command: dict = {
            "find": collection,
            "filter": spec["filter"],
            "sort": dict(spec["sort"]),
            "skip": spec["skip"],
            "limit": spec["limit"],
        }
    else:
        if endpoint == "employee_monthly":
            start, end = checked_month(_require(month, "month"))
            collection, pipeline = "attendance_logs", monthly_pipeline(_require(emp_code, "emp_code"), start, end)
        elif endpoint == "department_summary":
            start, end = checked_month(_require(month, "month"))
            collection, pipeline = "employees", summary_pipeline(start, end, department)
        elif endpoint == "late_leaderboard":
            start, end = checked_month(_require(month, "month"))
            collection, pipeline = "attendance_logs", leaderboard_pipeline(start, end, limit, department)
        else:  # department_trend
            dept = _require(department, "department")
            days = trend_range(_require(frm, "from"), _require(to, "to"))
            collection, pipeline = "employees", trend_pipeline(dept, frm, days)
        command = {"aggregate": collection, "pipeline": pipeline, "cursor": {}}
    result = db.command({"explain": command, "verbosity": "executionStats"})
    return {"endpoint": endpoint, "collection": collection, "explain": json.loads(json_util.dumps(result))}
