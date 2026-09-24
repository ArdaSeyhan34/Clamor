# Running Clamor on real feedback

This guide is for using Clamor inside a company, on real app-store reviews and support
tickets. The short version:

```bash
mkdir -p data/private                       # ignored by git, see .gitignore
# put the exports there, then:
clamor analyze data/private/play_reviews.csv data/private/tickets.csv \
    --releases data/private/releases.csv \
    --language tr --product-name "YourApp" --no-llm \
    --out data/private/report
```

Open `data/private/report/report.html`, or run `streamlit run app/streamlit_app.py` and use
*Upload your own* (*Kendi verini yükle* in Türkçe).

The report and briefs come out in the language of the feedback; add `--report-language en`
for an English report on Turkish feedback, or `--report-language tr` for the reverse.

## 1. Keep the data out of git

Everything under `data/private/` and every `*.private.csv` file is ignored by git, so it
cannot be committed or pushed by accident. Check before your first commit:

```bash
git check-ignore -v data/private/tickets.csv   # prints the .gitignore rule that matches
git status --short                             # the file must not be listed
```

Write reports to `data/private/` too (`--out data/private/report`): they quote customers.

## 2. Export the feedback

Clamor needs a text column and a date column. Everything else is optional and is picked up
by name, in English or Turkish.

| Source | How to export | What Clamor reads |
|---|---|---|
| Google Play | Play Console → *Download reports* → *Reviews*, one CSV per month (UTF-16) | `Review Text`, `Review Title`, `Review Submit Date and Time`, `Star Rating` |
| App Store | App Store Connect API (`customerReviews`) or any review tool's CSV export | `title`, `body` / `review`, `date`, `rating` |
| Support tickets | CSV or Excel export of the helpdesk or CRM | `Konu` / `Subject` + `Açıklama` / `Description`, `Oluşturma Tarihi` / `Created`, `Talep No` |
| Surveys | CSV export with the comment and the score | `Yorum` / `Comment`, `Tarih` / `Date`, `Puan` / `Score` |

The loader handles UTF-8, UTF-16 and Windows-1254 files, comma or semicolon separators
(Excel in a Turkish locale writes `;`), day-first dates such as `12.03.2026 10:15`, and
prepends a subject or review title to the body.

Pass several files at once and they are combined. Each file keeps its own channel (its
name without a trailing date, unless it has a `channel` column), so the report shows
whether a problem comes up in reviews, in tickets or in both. Monthly Play Console files
(`reviews_202601.csv`, `reviews_202602.csv`, ...) end up in one channel.

If the export has a customer or user ID column (`user_id`, `Müşteri No`, ...), reach counts
distinct people; without one, every item counts as its own person.

## 3. Add the release history

The release radar needs a small table of what shipped and when:

```csv
date;title;description
10.03.2026;v5.1.1 Login fix;SMS verification code arrives late or expires
02.04.2026;v5.2 Balance history;Monthly spending summary and transaction list
```

The description is what links a release to a theme, so describe the problem it addresses
in the words customers would use. Store release notes are usually good enough.

## 4. Privacy

- **Personal data is masked on load** (`--redact`, the default): e-mail addresses, phone
  numbers, card numbers that pass the Luhn check, IBANs and Turkish national ID numbers
  that pass their checksum. Masking happens before embedding, so the placeholders are
  what appears in themes, quotes and reports. Names, addresses and anything written in
  words are *not* masked, so treat the reports as confidential.
- **Everything runs locally.** Embedding models run on your CPU; the only download is
  the model itself, once.
- **Claude is off unless you turn it on.** The Claude layer sends theme keywords and
  representative quotes to the Anthropic API. It is used automatically when
  `ANTHROPIC_API_KEY` is set, so pass `--no-llm` (as above) until your company has
  approved sending customer text to an external API.
- **Do not deploy the dashboard publicly with real data.** Run `streamlit run` on your own
  machine; uploads are then processed in memory and never leave it.

## 5. Choosing the model

| `--backend` | When to use it |
|---|---|
| `hybrid` (default for `--language tr`) | Multilingual MiniLM plus word overlap: the purest Turkish themes in the benchmark, and robust to mixed Turkish/English text. Downloads ~470 MB once. |
| `multilingual` | The semantic model alone: fewer, broader themes, but it tends to merge distinct problems. |
| `tfidf` | No download at all, e.g. when the model host is blocked on a company network. Needs shared wording to group items. |

If the model cannot be downloaded, Clamor falls back to `tfidf` and says so.

Theme granularity is controlled by `--threshold` (the clustering distance threshold).
Raise it by 0.05 if one problem is split across several themes; lower it if unrelated
problems end up together. The per-backend defaults were tuned on the synthetic benchmarks.

## 6. Reading the results without revenue data

Consumer apps usually have no per-user revenue, so the ranking uses reach (how many
people), severity (share of negative mentions) and momentum (statistically significant
growth); the revenue weight is set to zero automatically. Presets still apply:
`--preset quality` favors painful and growing problems, `--preset growth` favors the
widest reach.

A useful routine for a weekly review:

1. Look at **early warnings** first: themes growing faster than chance, with the
   confidence interval and FDR-adjusted q-value.
2. Check the **release radar** for the last release: *Worse* next to a release is the
   fastest signal of a regression.
3. Read the **roadmap** top 10 with its quotes, and compare it to what the team believes
   the top problems are. The gaps are where the conversation is worth having.
4. Re-run with `--as-of` set to a past date to see what Clamor would have said then.

## 7. Sanity checks on a first run

- **Themes that are really greetings or signatures:** add the recurring phrase to the
  boilerplate prototypes of the language in `clamor/lang.py`.
- **The product or company name dominating keywords:** pass it with `--product-name`
  (repeatable).
- **Dates parsed wrongly:** the first line of the report shows the date range. The date
  order is read from the whole column: a day above 12 settles it, and otherwise dotted
  or dashed dates (`03.04.2026`) are read day first and slashed ones (`03/04/2026`) month
  first. Convert the column to ISO dates (`2026-04-03`) if in doubt.
- **Too few items:** statistics need volume. Below roughly 300 items or 8 weeks of
  history, early warnings will rarely be significant; theme discovery still works.
