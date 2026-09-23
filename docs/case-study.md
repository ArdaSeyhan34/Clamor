# Case study: what should Tempo build next quarter?

*A product memo written from Clamor's output on the demo dataset. Tempo is a fictional
B2B team-calendar SaaS; the data is simulated (see [methodology](methodology.md)), but
the reasoning is the same you would apply to a real export.*

---

## Context

- **Product:** Tempo, team scheduling and calendar sync for companies of 1 to 600 seats.
- **Customers:** 600 accounts, $187k MRR. 20 enterprise accounts hold 75% of revenue;
  319 free accounts hold none.
- **Feedback:** 3,170 items between January and early July 2026: app-store reviews, NPS
  comments, support tickets, community posts and sales-call notes.
- **What we shipped:** guided onboarding (v3.1), a Pro price increase, a rewritten sync
  engine (v3.2), a sync hotfix (v3.3), an Android performance pass (v3.4) and a redesigned
  recurring-events editor (v3.5).
- **The question:** the team has room for about three initiatives next quarter. Which
  ones?

## What the feedback says

Clamor found 25 themes (19 actionable, 6 praise). The top of the ranked roadmap, with the
default *balanced* weights:

| # | Theme | Mentions (60d) | Revenue-weighted MRR | Trend | # by votes |
|---:|---|---:|---:|---|---:|
| 1 | The mobile app takes forever to load | 190 | $9.3k | stable | 1 |
| 2 | Recurring event editor won't set custom repeats | 41 | $4.8k | **emerging, 3.3x** | 9 |
| 3 | Calendar sync is unreliable | 55 | $15.7k | declining | 6 |
| 4 | Meeting times wrong across time zones | 70 | $15.7k | stable | 5 |
| 5 | Schedule meetings from Slack | 133 | $20.9k | stable | 3 |
| 6 | Too many email notifications | 76 | $10.2k | stable | 4 |
| 7 | SSO / SAML | 32 | **$34.5k** | stable | 10 |
| ... | | | | | |
| 11 | Dark mode | 172 | $3.4k | stable | **2** |

The full report is in [`reports/demo/report.md`](../reports/demo/report.md).

### Finding 1: the v3.5 editor broke recurring meetings, and it is getting worse

Recurring-event complaints were a trickle (about 2 a week) until v3.5 shipped on June 11.
Since then their share of all feedback has tripled (3.3x, 95% CI 1.9-5.9, q < 0.001).
Clamor linked the release notes to this theme on its own and gave the release a
**Worse** verdict. Typical quotes: *"Editing one occurrence changes the whole recurring
series"*, *"The repeat every other week option is broken"*.

By raw count it is only the #9 topic, which is how regressions hide: they start small.

### Finding 2: our Android performance work did not land

v3.4 promised faster startup and smoother scrolling. The mobile-performance theme is still
our #1 complaint, and its mention rate after the release is statistically
indistinguishable from before (x1.08, 95% CI 0.79-1.47): **no detectable change**, with
86 mentions before and 84 after, so this is not a data problem. Either the fix does not
reach the devices people use, or it fixed the wrong thing. The quotes point at older
Android phones and the week view.

### Finding 3: SSO is the quietest request and the most expensive one

Only 32 mentions in 60 days, #10 by volume. But 69% of them come from enterprise accounts,
through sales calls and support tickets: *"We cannot renew without SAML SSO"*, *"Missing
SSO support is a blocker in our procurement review"*. Its revenue-weighted demand
($34.5k MRR) is the highest of any theme and 10x that of dark mode, and the accounts
raising it hold $1.6M ARR. A separate theme about SCIM provisioning ("offboarding users
by hand is a compliance risk") belongs to the same initiative. Under the *enterprise*
weight preset, SSO is the #1 priority.

### Finding 4: dark mode is loud, not valuable

Dark mode is the #2 most-mentioned request (172 mentions). 66% of them come from free
accounts, most often through app-store reviews, and the tone is mild ("would love a dark
theme"). It ranks #11. It is still a cheap win for app-store ratings, but not a
quarter-defining bet.

### Finding 5: the sync hotfix worked

v3.2's sync-engine rewrite caused the biggest incident of the half-year: sync complaints
went from 15 to 125 in the four weeks after release (x3.5, **Worse**). Clamor would have
flagged it as emerging a week after the release (see the time-travel screenshot in the
README). The v3.3 hotfix brought the rate down to x0.19 of the incident level
(**Resolved**). The theme is now declining and should leave the roadmap once the 60-day
window has passed.

## Recommendations for next quarter

| Priority | Initiative | Why | Success metric |
|---|---|---|---|
| **1** | **Fix the recurring-events editor now** (hotfix, not a roadmap item) | An active regression affecting 39 accounts and growing 3x; it touches the core scheduling flow | Recurring-events theme gets a *Resolved* verdict in the release radar within 4 weeks |
| **2** | **SSO (SAML + SCIM) for enterprise** | Highest revenue-weighted demand; explicit renewal and procurement blockers | Enterprise renewals with SSO requirements closed; SSO theme mentions from enterprise go to near zero |
| **3** | **Mobile performance, second attempt, diagnosis first** | #1 complaint by reach and severity, and the first attempt changed nothing | Mobile-performance mention rate x0.7 or lower after the release; app-store rating up |

**Not this quarter:**

- *Dark mode:* high volume, low value. Revisit as a small app-store-rating experiment.
- *Slack integration:* strong and broad demand ($20.9k revenue-weighted), but stable and
  not urgent. The best candidate for the following quarter.
- *Pricing complaints:* real, but they call for packaging decisions (inactive seats, a
  nonprofit tier) rather than engineering work. Handed to the pricing owner.

## What I would do before committing

1. **Talk to five SSO accounts** from the quotes to separate "must have for renewal" from
   "nice to have", and check the renewal calendar.
2. **Pull crash and startup-time telemetry** for Android, split by device age, to find
   out why v3.4 did not move the needle.
3. **Reproduce the recurring-events bugs** with the accounts behind the quotes, and add a
   regression test for series edits.

## Caveats

- The release radar is observational: it shows that complaints changed after a release,
  not that the release caused it. Here the timing and content make the link plausible.
- Feedback over-represents vocal users. Revenue weighting corrects for *who* is speaking,
  not for customers who never speak. Usage data should complement it.
- The theme names in this memo were shortened by hand. With an API key, Clamor's Claude
  review produces names like these automatically.
