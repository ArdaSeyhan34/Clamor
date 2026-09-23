"""Synthetic customer-feedback generator with known ground truth.

Real feedback datasets rarely come with labels, revenue data *and* a release history,
so Clamor ships with a simulator for **Tempo**, a fictional B2B team-calendar SaaS.

The simulation is deliberately built to contain the situations a product team gets
wrong when it just counts votes:

* a *loud but cheap* request (dark mode: many free users),
* a *quiet but expensive* request (SSO: a handful of enterprise accounts),
* a regression introduced by a release and fixed by a later one (calendar sync),
* a fix that did **not** move the needle (Android performance),
* a fresh regression that is still unfolding when the data ends (recurring events),
* a slow-burning annoyance (notification overload).

Every generated item carries a ground-truth theme and polarity in a separate table,
which is what makes the evaluation in :mod:`clamor.evaluate` possible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

START_DATE = pd.Timestamp("2026-01-05")  # a Monday
DEFAULT_DAYS = 182  # 26 weeks

PLANS: dict[str, dict] = {
    "free": {"share": 0.55, "seats": (1, 3), "price_per_seat": 0},
    "pro": {"share": 0.28, "seats": (2, 15), "price_per_seat": 12},
    "business": {"share": 0.13, "seats": (10, 80), "price_per_seat": 25},
    "enterprise": {"share": 0.04, "seats": (80, 600), "price_per_seat": 38},
}

PLAN_CHANNELS: dict[str, dict[str, float]] = {
    "free": {"app_store": 0.45, "community": 0.25, "nps_survey": 0.20, "support_ticket": 0.10},
    "pro": {"app_store": 0.25, "community": 0.15, "nps_survey": 0.25, "support_ticket": 0.35},
    "business": {
        "app_store": 0.05,
        "community": 0.10,
        "nps_survey": 0.25,
        "support_ticket": 0.45,
        "sales_call": 0.15,
    },
    "enterprise": {"nps_survey": 0.20, "support_ticket": 0.45, "sales_call": 0.35},
}

REGIONS = ["North America", "Europe", "APAC", "LATAM"]
INDUSTRIES = ["Software", "Agency", "Education", "Healthcare", "Finance", "Retail", "Nonprofit"]


@dataclass(frozen=True)
class ThemeSpec:
    key: str
    kind: str  # bug | feature_request | ux | pricing | praise
    polarity: str  # negative | neutral | positive
    base_rate: float  # expected items per week before any event
    plan_affinity: dict[str, float]
    phrases: tuple[str, ...]
    channel_boost: dict[str, float] = field(default_factory=dict)
    weekly_growth: float = 0.0  # linear growth of the base rate per week


@dataclass(frozen=True)
class Release:
    day: int  # offset from START_DATE
    version: str
    title: str
    description: str
    theme_hint: str  # the theme the team *intended* to affect


@dataclass(frozen=True)
class Event:
    """A multiplicative change to a theme's rate, caused by a release."""

    theme: str
    start_day: int
    multiplier: float
    end_day: int | None = None
    ramp_days: int = 3  # how fast the new rate kicks in (users need time to notice)


SLOTS: dict[str, list[str]] = {
    "provider": ["Google Calendar", "Outlook", "Office 365", "iCloud", "Google"],
    "since": ["since the update", "since Tuesday", "again", "this week", "randomly", "for days"],
    "when": ["every morning", "every few hours", "whenever I reconnect", "all the time"],
    "platform": ["Android", "iOS", "mobile", "Android"],
    "device": ["Pixel", "Samsung phone", "iPhone", "older Android phone", "tablet"],
    "idp": ["Okta", "Azure AD", "Entra ID", "OneLogin", "Google Workspace"],
    "tz": ["UTC", "Pacific time", "CET", "GMT", "Eastern time"],
    "city": ["London", "Berlin", "Singapore", "New York", "Sao Paulo", "Sydney"],
    "chat": ["Slack", "Microsoft Teams", "Slack", "Teams"],
    "nth": ["second", "third", "fourth", "fifth"],
    "n": ["5", "12", "25", "40", "80", "150"],
    "months": ["3", "6", "9", "18", "24"],
    "weeks": ["2", "3", "6", "8"],
    "name": ["Maria", "Deniz", "Chen", "Priya", "Lukas", "Amara", "Jonas", "Sofia", "Omar", "Kate"],
    "role": [
        "the IT director",
        "their CTO",
        "the VP of Operations",
        "the security lead",
        "the office manager",
        "the head of RevOps",
    ],
}

THEMES: tuple[ThemeSpec, ...] = (
    ThemeSpec(
        "calendar_sync",
        "bug",
        "negative",
        4.0,
        {"free": 0.35, "pro": 0.30, "business": 0.20, "enterprise": 0.15},
        (
            "my {provider} calendar stopped syncing {since}",
            "every event from {provider} now shows up twice",
            "meetings I create in Tempo never make it to {provider}",
            "two-way sync with {provider} is broken and changes take hours to appear",
            "sync keeps creating duplicate events on my calendar",
            "deleted meetings keep coming back after every sync",
            "{provider} sync fails with an error {when}",
            "we missed a client call because the invite never synced to {provider}",
            "half of my {provider} events disappeared from Tempo",
            "calendar sync is completely unreliable {since}",
            "the sync status is stuck on syncing forever",
            "reconnecting {provider} does not fix the sync problem",
        ),
        {"support_ticket": 1.5},
    ),
    ThemeSpec(
        "mobile_performance",
        "bug",
        "negative",
        14.0,
        {"free": 0.55, "pro": 0.30, "business": 0.12, "enterprise": 0.03},
        (
            "the {platform} app takes forever to load",
            "the {platform} app crashes when I open the week view",
            "scrolling the calendar on {platform} is laggy and slow",
            "the app freezes every time I switch calendars on my {device}",
            "the mobile app drains my battery",
            "the {platform} app is so slow it is unusable on older phones",
            "it keeps crashing on startup after the last {platform} update",
            "the app hangs for ten seconds before showing my schedule",
            "tapping an event freezes the {platform} app",
            "performance on my {device} is terrible compared to the web version",
        ),
        {"app_store": 1.6},
    ),
    ThemeSpec(
        "dark_mode",
        "feature_request",
        "neutral",
        16.0,
        {"free": 0.65, "pro": 0.27, "business": 0.07, "enterprise": 0.01},
        (
            "please add a dark mode",
            "would love a dark theme for late night planning",
            "dark mode please, the white screen hurts my eyes",
            "any plans for a dark theme? every other app has one",
            "the bright interface is painful at night, we need a dark mode",
            "add a night mode option to the {platform} app",
            "dark mode is the only thing missing for me",
            "please follow the system dark theme on {platform}",
            "I would honestly pay extra for a dark mode",
        ),
        {"app_store": 1.4, "community": 1.4},
    ),
    ThemeSpec(
        "sso_saml",
        "feature_request",
        "neutral",
        4.0,
        {"free": 0.0, "pro": 0.0, "business": 0.25, "enterprise": 0.75},
        (
            "we need SAML single sign-on before we can roll out to more teams",
            "our security team requires SSO with {idp}",
            "missing SSO support is a blocker in our procurement review",
            "please add SCIM provisioning so IT can manage users from {idp}",
            "we cannot renew without SAML SSO and enforced login through {idp}",
            "IT wants to manage access through {idp}, is SSO on the roadmap",
            "offboarding users by hand is a compliance risk, we need SCIM",
            "single sign-on with {idp} is a hard requirement for us",
        ),
        {"sales_call": 3.0},
    ),
    ThemeSpec(
        "pricing",
        "pricing",
        "negative",
        5.0,
        {"free": 0.15, "pro": 0.55, "business": 0.25, "enterprise": 0.05},
        (
            "the new per-seat price is too expensive for a small team",
            "the price increase does not feel justified",
            "we are considering switching because Tempo got too pricey",
            "paying per seat makes no sense for occasional users",
            "a 20 percent price jump with no new features is frustrating",
            "hard to justify the cost to my manager after the price increase",
            "please offer a cheaper plan for nonprofits",
            "the Pro plan now costs more than tools that do much more",
            "billing us for inactive seats is unfair",
        ),
        {"sales_call": 2.0, "nps_survey": 1.5},
    ),
    ThemeSpec(
        "timezone",
        "ux",
        "negative",
        7.0,
        {"free": 0.25, "pro": 0.35, "business": 0.28, "enterprise": 0.12},
        (
            "meeting times are wrong when attendees are in different time zones",
            "the scheduler keeps showing {tz} instead of my local time",
            "events shift by an hour after daylight saving time",
            "it is confusing which time zone the booking link uses",
            "invites to our {city} office arrive with the wrong time",
            "there is no way to see two time zones side by side when scheduling",
            "my time zone setting resets every time I travel",
        ),
    ),
    ThemeSpec(
        "slack_integration",
        "feature_request",
        "neutral",
        8.0,
        {"free": 0.20, "pro": 0.40, "business": 0.32, "enterprise": 0.08},
        (
            "please add a {chat} integration so reminders show up in our channels",
            "we would love to schedule meetings directly from {chat}",
            "a {chat} bot that posts the daily agenda would be amazing",
            "an integration with {chat} is the main thing missing for our team",
            "set my {chat} status automatically when I am in a meeting",
            "we live in {chat}, getting alerts there would help a lot",
            "can you connect Tempo with {chat} for meeting links",
        ),
        {"sales_call": 1.5},
    ),
    ThemeSpec(
        "notifications",
        "ux",
        "negative",
        4.0,
        {"free": 0.35, "pro": 0.35, "business": 0.20, "enterprise": 0.10},
        (
            "way too many email notifications, I am drowning in reminders",
            "I get three reminders for every single meeting",
            "I cannot turn off the daily digest emails",
            "notification settings are too coarse, it is all or nothing",
            "push notifications arrive late or twice",
            "every calendar change triggers an email to the whole team",
            "please let me mute notifications for specific calendars",
            "the reminder spam is making people ignore Tempo entirely",
        ),
        weekly_growth=0.06,
    ),
    ThemeSpec(
        "export_reporting",
        "feature_request",
        "neutral",
        6.0,
        {"free": 0.05, "pro": 0.30, "business": 0.50, "enterprise": 0.15},
        (
            "we need to export meeting data to CSV",
            "I would like a report of how much time the team spends in meetings",
            "please add an API endpoint for exporting events",
            "our finance team needs a monthly utilization report",
            "can we get analytics on meeting load per person",
            "export to Excel would save me hours every month",
            "a dashboard showing meeting hours by team would be great",
        ),
        {"sales_call": 2.0},
    ),
    ThemeSpec(
        "onboarding",
        "ux",
        "negative",
        10.0,
        {"free": 0.55, "pro": 0.30, "business": 0.12, "enterprise": 0.03},
        (
            "setting up the workspace was confusing",
            "it took me an hour of setup to figure out how to invite my team",
            "the first-time setup does not explain how anything works",
            "our new hires get lost in the onboarding flow",
            "the onboarding never explains the difference between workspaces and teams",
            "no guidance during setup, I did not know where to start",
            "the setup wizard skipped steps and left my workspace half configured",
            "onboarding is confusing for anyone who is not technical",
        ),
    ),
    ThemeSpec(
        "recurring_events",
        "bug",
        "negative",
        1.5,
        {"free": 0.30, "pro": 0.35, "business": 0.25, "enterprise": 0.10},
        (
            "editing one occurrence changes the whole recurring series",
            "weekly recurring meetings disappear after the {nth} week",
            "the new recurring event editor will not let me set custom repeats",
            "changing one recurring meeting creates extra copies of the whole series",
            "I cannot delete a single instance of a repeating event anymore",
            "recurring standups now show up on the wrong weekday after the redesign",
            "the repeat every other week option is broken",
            "exceptions to recurring events are lost when I save",
        ),
        {"support_ticket": 1.4},
    ),
    ThemeSpec(
        "praise",
        "praise",
        "positive",
        12.0,
        {"free": 0.45, "pro": 0.35, "business": 0.15, "enterprise": 0.05},
        (
            "love how easy it is to find a time that works for everyone",
            "best scheduling tool we have used",
            "the team availability view is fantastic",
            "Tempo saves me hours every week",
            "super clean interface and very intuitive",
            "our whole company switched and nobody looked back",
            "great product and the support team is very responsive",
            "the booking links alone are worth it",
            "simple, fast and reliable, exactly what we needed",
        ),
        {"sales_call": 0.3, "support_ticket": 0.4},
    ),
)

RELEASES: tuple[Release, ...] = (
    Release(
        38,
        "v3.1",
        "Guided onboarding",
        "A new setup wizard and checklist that walks new workspaces through inviting the "
        "team, so first-time setup is no longer confusing.",
        "onboarding",
    ),
    Release(
        52,
        "pricing-2026-02",
        "Pro plan price update",
        "The Pro plan price per seat increases from $10 to $12 per month.",
        "pricing",
    ),
    Release(
        66,
        "v3.2",
        "Sync engine 2.0",
        "Rebuilt the Google Calendar and Outlook sync engine for faster two-way sync.",
        "calendar_sync",
    ),
    Release(
        101,
        "v3.3",
        "Sync hotfix",
        "Fixes duplicate events and events that failed to sync with Google Calendar and Outlook.",
        "calendar_sync",
    ),
    Release(
        129,
        "v3.4",
        "Android performance pass",
        "Faster startup, less lag and smoother scrolling in the Android app.",
        "mobile_performance",
    ),
    Release(
        157,
        "v3.5",
        "Recurring events editor",
        "Redesigned editor for recurring meetings with custom repeat rules.",
        "recurring_events",
    ),
)

EVENTS: tuple[Event, ...] = (
    Event("onboarding", 38, 0.35, ramp_days=10),
    Event("pricing", 52, 2.6, ramp_days=2),
    Event("calendar_sync", 67, 7.0, end_day=102, ramp_days=2),
    Event("mobile_performance", 129, 1.0),  # the fix that did not move the needle
    Event("recurring_events", 158, 7.0, ramp_days=3),
)

CLOSERS = {
    "negative": [
        "Please fix this asap.",
        "This is really frustrating.",
        "Very annoying.",
        "Hope this gets fixed soon.",
        "Not happy about this.",
        "",
        "",
    ],
    "neutral": [
        "Thanks!",
        "Would really appreciate it.",
        "Is this on the roadmap?",
        "Thanks in advance.",
        "",
        "",
    ],
    "positive": ["Keep up the great work!", "Thank you!", "Five stars.", "", ""],
}
CONTEXT = [
    "We use Tempo every day.",
    "Our team of {n} people relies on Tempo.",
    "I have been a customer for {months} months.",
    "Tempo is central to how we plan our week.",
    "",
    "",
    "",
]
GREETINGS = ["Hi team,", "Hello,", "Hi Tempo support,", "Hey there,", ""]
SIGNOFFS = ["Thanks, {name}", "- {name}", "Best, {name}", ""]
DEAL_NOTES = [
    "Renewal is in {weeks} weeks.",
    "Expansion to {n} more seats depends on this.",
    "They mentioned evaluating competitors.",
    "Otherwise very happy with the product.",
    "",
]
COMMUNITY_PREFIX = {
    "bug": ["[Bug] ", "Anyone else seeing this? ", "Bug report: ", ""],
    "feature_request": ["Feature request: ", "Idea: ", "+1 for this: ", ""],
    "ux": ["Feedback: ", "Small thing, but ", ""],
    "pricing": ["Pricing feedback: ", ""],
    "praise": ["Shoutout: ", ""],
}

_WORD = re.compile(r"[A-Za-z]{5,}")


@dataclass
class SyntheticDataset:
    """All tables of one simulation. Only the first three are inputs to Clamor."""

    feedback: pd.DataFrame
    accounts: pd.DataFrame
    releases: pd.DataFrame
    ground_truth: pd.DataFrame  # feedback_id -> true theme and polarity
    events: pd.DataFrame  # what really happened to each theme's rate, and when
    release_truth: pd.DataFrame  # which theme each release was meant to change

    FILES = ("feedback", "accounts", "releases", "ground_truth", "events", "release_truth")

    def save(self, directory: str | Path) -> None:
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        for name in self.FILES:
            getattr(self, name).to_csv(out / f"{name}.csv", index=False)

    @classmethod
    def load(cls, directory: str | Path) -> SyntheticDataset:
        src = Path(directory)
        tables = {name: pd.read_csv(src / f"{name}.csv") for name in cls.FILES}
        tables["feedback"]["created_at"] = pd.to_datetime(tables["feedback"]["created_at"])
        return cls(**tables)


def _fill(template: str, rng: np.random.Generator) -> str:
    def repl(match: re.Match) -> str:
        options = SLOTS[match.group(1)]
        return options[rng.integers(len(options))]

    return re.sub(r"\{(\w+)\}", repl, template)


def _pick(options: list[str], rng: np.random.Generator) -> str:
    return options[rng.integers(len(options))]


def _typo(text: str, rng: np.random.Generator) -> str:
    words = list(_WORD.finditer(text))
    if not words:
        return text
    m = words[rng.integers(len(words))]
    i = m.start() + int(rng.integers(1, len(m.group()) - 2))
    return text[:i] + text[i + 1] + text[i] + text[i + 2 :]


def _sentence(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    if not text[1:2].isupper():  # keep "iCloud", "iOS" as they are
        text = text[0].upper() + text[1:]
    return text if text[-1] in ".!?" else text + "."


def _make_accounts(n_accounts: int, rng: np.random.Generator) -> pd.DataFrame:
    adjectives = [
        "Blue",
        "North",
        "Bright",
        "Silver",
        "Quiet",
        "Rapid",
        "Green",
        "Lucky",
        "Bold",
        "Clear",
        "Urban",
        "Summit",
        "Harbor",
        "Maple",
        "Iron",
        "Nova",
    ]
    nouns = [
        "Labs",
        "Studio",
        "Health",
        "Logistics",
        "Ventures",
        "Digital",
        "Partners",
        "Works",
        "Analytics",
        "Foods",
        "Robotics",
        "Media",
        "Capital",
        "Systems",
    ]
    plans = list(PLANS)
    shares = np.array([PLANS[p]["share"] for p in plans])
    plan_draw = rng.choice(plans, size=n_accounts, p=shares / shares.sum())
    rows = []
    used: set[str] = set()
    for i, plan in enumerate(plan_draw):
        lo, hi = PLANS[plan]["seats"]
        # heavy tail inside each plan: most accounts are small
        seats = int(np.clip(round(lo * (hi / lo) ** rng.beta(1.2, 3.0)), lo, hi))
        mrr = seats * PLANS[plan]["price_per_seat"]
        name = f"{_pick(adjectives, rng)} {_pick(nouns, rng)}"
        while name in used:
            name = f"{_pick(adjectives, rng)} {_pick(nouns, rng)} {rng.integers(2, 99)}"
        used.add(name)
        rows.append(
            {
                "account_id": f"ACC-{i + 1:04d}",
                "company": name,
                "plan": plan,
                "seats": seats,
                "mrr": float(mrr),
                "region": _pick(REGIONS, rng),
                "industry": _pick(INDUSTRIES, rng),
            }
        )
    return pd.DataFrame(rows)


def _rate_multiplier(theme: ThemeSpec, day: int) -> float:
    mult = 1.0 + theme.weekly_growth * (day / 7.0)
    for ev in EVENTS:
        if ev.theme != theme.key or day < ev.start_day:
            continue
        if ev.end_day is not None and day >= ev.end_day:
            continue
        progress = min(1.0, (day - ev.start_day + 1) / max(ev.ramp_days, 1))
        mult *= 1.0 + (ev.multiplier - 1.0) * progress
    return mult


def _render(
    theme: ThemeSpec,
    channel: str,
    account: pd.Series,
    rng: np.random.Generator,
) -> tuple[str, float | None]:
    phrase = _fill(_pick(list(theme.phrases), rng), rng)
    # 12% of items also mention a second, unrelated issue: real feedback is messy
    extra = ""
    if rng.random() < 0.12:
        other = THEMES[rng.integers(len(THEMES))]
        if other.key != theme.key and other.key != "praise":
            extra = " Also, " + _fill(_pick(list(other.phrases), rng), rng) + "."
    closer = _pick(CLOSERS[theme.polarity], rng)
    rating: float | None = None

    if channel == "app_store":
        stars = {"negative": [1, 1, 2, 2, 3], "neutral": [3, 4, 4, 5], "positive": [4, 5, 5]}
        rating = float(_pick(stars[theme.polarity], rng))
        text = f"{_sentence(phrase)} {closer}"
    elif channel == "nps_survey":
        scores = {"negative": range(0, 7), "neutral": range(5, 9), "positive": range(8, 11)}
        rating = float(_pick(list(scores[theme.polarity]), rng))
        text = f"{_sentence(phrase)}{extra}"
    elif channel == "support_ticket":
        parts = [
            _pick(GREETINGS, rng),
            _fill(_pick(CONTEXT, rng), rng),
            _sentence(phrase) + extra,
            closer,
            _fill(_pick(SIGNOFFS, rng), rng),
        ]
        text = " ".join(p for p in parts if p)
    elif channel == "sales_call":
        role = _pick(SLOTS["role"], rng)
        deal = _fill(_pick(DEAL_NOTES, rng), rng)
        text = f'Call notes ({account["company"]}): {role} said "{_sentence(phrase)}" {deal}'
    else:  # community
        prefix = _pick(COMMUNITY_PREFIX[theme.kind], rng)
        text = f"{prefix}{_sentence(phrase)}{extra} {closer}"

    text = re.sub(r"\s+", " ", text).strip()
    if rng.random() < 0.15:
        text = _typo(text, rng)
    if channel in {"app_store", "community"} and rng.random() < 0.25:
        text = text.lower()
    return text, rating


def generate(
    seed: int = 7,
    n_accounts: int = 600,
    days: int = DEFAULT_DAYS,
    volume_scale: float = 1.25,
) -> SyntheticDataset:
    """Simulate `days` of feedback for Tempo. Deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    accounts = _make_accounts(n_accounts, rng)
    by_plan = {p: accounts[accounts["plan"] == p] for p in PLANS}
    # bigger accounts talk to you more often, but not linearly more often
    plan_weights = {
        p: np.sqrt(df["seats"].to_numpy(float)) / np.sqrt(df["seats"].to_numpy(float)).sum()
        for p, df in by_plan.items()
    }
    plans = list(PLANS)

    feedback_rows, truth_rows = [], []
    for day in range(days):
        date = START_DATE + pd.Timedelta(days=day)
        weekday_factor = 0.55 if date.weekday() >= 5 else 1.12
        for theme in THEMES:
            lam = theme.base_rate / 7.0 * _rate_multiplier(theme, day) * weekday_factor
            for _ in range(rng.poisson(lam * volume_scale)):
                aff = np.array([theme.plan_affinity.get(p, 0.0) for p in plans])
                plan = plans[rng.choice(len(plans), p=aff / aff.sum())]
                acc_df = by_plan[plan]
                account = acc_df.iloc[rng.choice(len(acc_df), p=plan_weights[plan])]
                ch_mix = PLAN_CHANNELS[plan]
                channels = list(ch_mix)
                w = np.array([ch_mix[c] * theme.channel_boost.get(c, 1.0) for c in channels])
                channel = channels[rng.choice(len(channels), p=w / w.sum())]
                text, rating = _render(theme, channel, account, rng)
                ts = date + pd.Timedelta(seconds=int(rng.integers(7 * 3600, 22 * 3600)))
                fid = f"FB-{len(feedback_rows) + 1:05d}"
                feedback_rows.append(
                    {
                        "feedback_id": fid,
                        "created_at": ts,
                        "channel": channel,
                        "account_id": account["account_id"],
                        "rating": rating,
                        "text": text,
                    }
                )
                truth_rows.append(
                    {"feedback_id": fid, "true_theme": theme.key, "true_polarity": theme.polarity}
                )

    feedback = pd.DataFrame(feedback_rows).sort_values("created_at", kind="stable")
    feedback = feedback.reset_index(drop=True)
    releases = pd.DataFrame(
        [
            {
                "date": (START_DATE + pd.Timedelta(days=r.day)).date().isoformat(),
                "version": r.version,
                "title": r.title,
                "description": r.description,
            }
            for r in RELEASES
        ]
    )
    release_truth = pd.DataFrame(
        [{"version": r.version, "intended_theme": r.theme_hint} for r in RELEASES]
    )
    events = pd.DataFrame(
        [
            {
                "theme": e.theme,
                "start_date": (START_DATE + pd.Timedelta(days=e.start_day)).date().isoformat(),
                "end_date": None
                if e.end_day is None
                else (START_DATE + pd.Timedelta(days=e.end_day)).date().isoformat(),
                "multiplier": e.multiplier,
                "kind": "spike" if e.multiplier > 1 else ("drop" if e.multiplier < 1 else "none"),
            }
            for e in EVENTS
        ]
        + [
            {
                "theme": t.key,
                "start_date": START_DATE.date().isoformat(),
                "end_date": None,
                "multiplier": float("nan"),
                "kind": "gradual growth",
            }
            for t in THEMES
            if t.weekly_growth > 0
        ]
    )
    return SyntheticDataset(
        feedback=feedback,
        accounts=accounts,
        releases=releases,
        ground_truth=pd.DataFrame(truth_rows),
        events=events,
        release_truth=release_truth,
    )
