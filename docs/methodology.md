# Methodology

This document explains every modeling and statistical choice in Clamor, what alternatives
were tried and how they were measured. All numbers come from the demo dataset (seed 7,
3,170 items) and can be reproduced with `clamor evaluate` and `python scripts/ablation.py`.

- [1. The benchmark: a simulator with ground truth](#1-the-benchmark-a-simulator-with-ground-truth)
- [2. From tickets to segments](#2-from-tickets-to-segments)
- [3. Embeddings](#3-embeddings)
- [4. Clustering](#4-clustering)
- [5. Describing themes](#5-describing-themes)
- [6. Sentiment](#6-sentiment)
- [7. Opportunity score](#7-opportunity-score)
- [8. Early warning](#8-early-warning)
- [9. Release radar](#9-release-radar)
- [10. Evaluation protocol](#10-evaluation-protocol)
- [11. What did not work](#11-what-did-not-work)

---

## 1. The benchmark: a simulator with ground truth

Public feedback datasets rarely have revenue data, a changelog *and* labels at the same
time, and without labels an unsupervised pipeline cannot be evaluated. So
`clamor/synth.py` simulates **Tempo**, a fictional B2B team-calendar SaaS:

- **600 accounts** on four plans (free, pro, business, enterprise) with seat counts drawn
  from heavy-tailed distributions and MRR = seats x plan price ($187k MRR in total, 75%
  of it from 20 enterprise accounts).
- **12 topics** with different base rates, plan affinities and channels. SSO requests
  come mostly from enterprise accounts via sales calls; dark-mode requests come from free
  users via app-store reviews.
- **Five channels** with their own templates: app-store reviews (with stars), NPS
  comments (with scores), support tickets (greetings, context, sign-offs), sales-call
  notes (the customer's quote inside an account manager's note) and community posts
  (forum tags like `[Bug]`).
- **Noise on purpose:** 15% of items contain a typo, 25% of app-store and forum posts are
  lower-cased, about one item in ten mentions a second, unrelated topic, and generic sentences
  ("We use Tempo every day", "Please fix this asap") are shared across topics.
- **Six releases and five events:** onboarding improves after v3.1, pricing complaints
  jump after a price change, sync complaints explode after v3.2 and drop after the v3.3
  hotfix, the Android performance pass does nothing, and the v3.5 recurring-events editor
  introduces a regression that is still unfolding when the data ends. Notification
  complaints grow slowly throughout.

Ground truth (true topic and polarity of every item, event dates, the theme each release
was meant to change) is stored in separate files that the pipeline never reads.

## 2. From tickets to segments

**Problem.** Embedding whole tickets clusters them by *writing style*. Every support
ticket starts with "Hi team" and ends with "Thanks, Maria"; every sales note starts with
"Call notes (Acme): the CTO said". In early experiments, one cluster collected 440
items, mostly support tickets, spanning all twelve topics.

**Approach** (`clamor/text.py`):

1. If an item quotes the customer (`"..."`), keep only the quotes. The rest of a sales
   note is the account manager talking.
2. Split into sentences. Strip forum tags (`[Bug]`, `Feature request:`), greetings and
   discourse markers ("Also,", "Small thing, but"), and drop segments that are only a
   sign-off ("Thanks, Maria", "- Omar"). Sign-off names must be capitalized, so "Best app
   ever!" survives.
3. Mark segments as **boilerplate** when their embedding is within cosine 0.6 of one of
   ~25 prototype sentences ("Please fix this asap", "I have been a customer for 2
   years"). Matching is semantic, so the list does not have to be exhaustive.
4. Cluster the remaining **content segments**. An item's first content segment defines
   its *primary* theme, and every theme it mentions counts as a mention.

Working at sentence level also fixes the multi-topic problem: "Please add dark mode.
Also, onboarding was confusing." counts toward both themes instead of landing halfway
between them.

| Variant | ARI | Homogeneity |
|---|---:|---:|
| Sentence segments + boilerplate filter (default) | **0.815** | **0.953** |
| Sentence segments, no boilerplate filter | 0.612 | 0.804 |
| Whole tickets | 0.415 | 0.637 |

## 3. Embeddings

| Backend | ARI | NMI | Homogeneity | Items recovered | Notes |
|---|---:|---:|---:|---:|---|
| TF-IDF (word + char n-grams) + LSA | 0.445 | 0.691 | 0.864 | 82% | no download; purely lexical |
| **MiniLM (all-MiniLM-L6-v2, ONNX)** | **0.815** | **0.872** | 0.953 | 93% | default |
| Hybrid: MiniLM 0.8 + TF-IDF 0.2 | 0.796 | 0.866 | **0.957** | **94%** | on par |

- **Why MiniLM:** it is small (22M parameters), fast on CPU and good at short
  sentences. Running it through **ONNX Runtime** instead of PyTorch keeps the install
  light (a ~15 MB runtime instead of ~800 MB) and makes it deployable on free hosting.
- **Why not TF-IDF:** lexical similarity does not know that "the white screen hurts my
  eyes at night" and "please add a dark mode" are the same request. It splits themes by
  phrasing (48 themes instead of about 25).
- **The hybrid story:** before theme consolidation existed, MiniLM alone merged
  "recurring events editor is broken" into the semantically close "sync creates duplicate
  events", and adding a lexical component fixed that. Once consolidation was added (and
  the MiniLM threshold re-tuned), plain MiniLM recovered the recurring-events theme just
  as well. The benchmark could not separate the two, so the simpler model became the
  default. The hybrid remains available (`--backend hybrid`).

## 4. Clustering

**Agglomerative clustering, average linkage, cosine distance threshold 0.65**
(`clamor/themes.py`).

- **Why not k-means:** the number of themes is what we are looking for, and feedback
  themes are extremely unbalanced (480 dark-mode items next to 76 recurring-events
  items). K-means prefers equal-sized clusters: in experiments it split dark mode in
  two and merged the small themes, and the silhouette score kept rising with *k*
  instead of pointing to a sensible number of themes.
- **Why a distance threshold:** it has a meaning a product person can reason about ("how
  similar must two comments be to count as the same request?") and it does not force a
  number of themes.
- **Small clusters** (< 0.5% of items, minimum 5) are dissolved: their members join the
  closest large theme if the similarity is at least 0.5, otherwise they stay unassigned.
- **Scale:** agglomerative clustering is quadratic, so above 6,000 segments Clamor first
  compresses them into micro-clusters with k-means and clusters the centroids.

**Consolidation.** A strict threshold gives very pure themes (homogeneity 0.95) but
sometimes splits one request into two ("SSO with Okta" and "we need SAML single sign-on").
Neither signal alone is a safe merge rule:

- by meaning alone, "Slack integration" sits closer to "email notifications" (cosine
  0.58) than the two SSO halves sit to each other (0.57);
- by vocabulary alone, unrelated complaints share generic words.

So two themes merge only if **both** the centroid similarity (>= 0.45) **and** the
c-TF-IDF vocabulary similarity (>= 0.18) agree, with *complete linkage* (every pair
across two groups must qualify). An earlier transitive version chained "export to Excel
would save me hours" into the praise theme "saves me hours every week"; complete linkage
prevents that. Consolidation raises ARI from 0.771 to 0.815 and cuts spurious trend
flags from 4 to 1.

The remaining over-splitting is mostly harmless or even useful: "per-seat pricing is
unfair" and "the price increase is not justified" are both pricing, but they call for
different fixes. The optional Claude review decides such merges with actual judgment.

## 5. Describing themes

- **Keywords:** class-based TF-IDF (c-TF-IDF): each theme's segments form one document,
  so the keywords are the terms that are distinctive *for this theme*. Bigrams get a 1.6x
  boost because "dark mode" says more than "dark" and "mode".
- **Name (offline):** the most central short segment, i.e. a real customer sentence such
  as "Offboarding users by hand is a compliance risk, we need SCIM". Keyword-soup names
  ("users hand · risk need") were tried first and were unreadable.
- **Type:** bug, feature request, usability, pricing or praise, from the share of
  segments containing cue phrases ("crashes", "please add", "price") plus mean
  sentiment. Praise themes are kept but excluded from the roadmap.
- **Name and type (with Claude):** see the README; Claude sees keywords, statistics and
  8 representative quotes per theme and returns structured JSON.

## 6. Sentiment

A compact domain lexicon (~100 terms) with negation ("not great") and intensifiers
("really slow"), squashed to [-1, 1] as in VADER. General-purpose lexicons miss software
vocabulary: "crashes", "laggy", "duplicate", "stuck" are neutral in a movie review. Where
a channel carries a rating, text sentiment is blended with it (60/40): stars map 3 to 0,
NPS maps 7 to 0.

Sign accuracy on items whose true polarity is positive or negative: **87%** (positive
99%, negative 84%). Neutral-sounding complaints ("I get three reminders for every
meeting") are the typical miss, which is why severity is measured at theme level
(share of clearly negative mentions) rather than trusted item by item.

## 7. Opportunity score

For each actionable theme, over the last 60 days:

| Component | Definition | Scaling |
|---|---|---|
| Reach | distinct accounts mentioning the theme | sqrt(x / max) |
| Revenue | **revenue-weighted demand**: sum over accounts of MRR x (mentions of this theme / all mentions by that account) | sqrt(x / max) |
| Severity | share of mentions with sentiment < -0.2 | as is |
| Momentum | log2(rate ratio) / 2, clipped to [0, 1], **only if** the trend test is significant | as is |

`score = 100 x (w_reach x reach + w_revenue x revenue + w_severity x severity + w_momentum x momentum)`,
with weights normalized to sum to 1. Presets: *balanced* (25/35/20/20), *growth*,
*enterprise*, *quality*.

Design notes:

- **Revenue-weighted demand instead of "MRR of accounts that mentioned it".** The naive
  version saturates: every theme is eventually mentioned by some enterprise account, so
  every theme looked like $100k+ MRR. Splitting each account's MRR across its mentions
  measures what share of that customer's attention the theme holds (SSO: $34.5k, dark
  mode: $3.4k).
- **Square root instead of log or linear scaling.** Log scaling compressed every theme
  into 0.6-1.0, while linear scaling let the top theme flatten everything else.
- **Momentum requires significance.** A theme that went from 2 to 5 mentions has a lift
  of 2.5 and means nothing.
- **A 60-day window** makes the roadmap reflect current demand. A resolved incident drops
  down the list instead of dominating it for months.
- Every score is decomposed into points per component (the stacked bars in the report),
  and `rank_shift` shows how far it moved from a pure vote count.

## 8. Early warning

For an as-of date, each theme's mentions in the **recent** window (28 days) are compared
with the **baseline** (the 84 days before).

**Test.** Given `n = a + b` mentions across both windows and the expected ratio `f`
under "nothing changed", `a ~ Binomial(n, f / (1 + f))`. This is the exact conditional
test for the ratio of two Poisson rates. It stays valid for small counts, which is
exactly when an early warning matters. The Clopper-Pearson interval on the binomial
proportion converts to a 95% interval for the rate ratio.

**Normalization: median-of-ratios.** The expected ratio `f` cannot simply be total
recent volume over total baseline volume: when one theme explodes (the sync incident
reached 30% of all feedback), the total inflates and every *other* theme appears to
decline. Borrowing from RNA-seq differential expression (DESeq2), `f` is the median
across themes of `a_t / b_t`, because most themes do not change between two windows.
In the backtest this cuts spurious rising/declining flags from 12 theme-days to 1 with no
loss of detection.

**Multiple testing.** Twenty-five themes are tested every time. Benjamini-Hochberg keeps
the false discovery rate at 5%.

**Statuses:** *new* (significant, no baseline mentions), *emerging* (q < 0.05 and ratio
>= 1.5), *rising*, *declining*, *stable*, *quiet* (fewer than 5 mentions in total).

## 9. Release radar

1. **Match.** "title. description" of each release is embedded with the same model and
   linked to the most similar theme centroid (if the similarity is at least 0.35).
   6/6 releases were linked to the right theme.
2. **Windows.** Up to 28 days before and after the release, **cut at neighbouring
   releases** so that the effect of v3.2 is not credited to v3.1.
3. **Test.** The same exact rate test and median-of-ratios normalization as above.
4. **Verdicts:** *Resolved* (significant, ratio <= 0.5), *Improved* (significant,
   < 1), *Worse* (significant, > 1), *No detectable change* (not significant, reasonably
   tight interval), *Inconclusive* (fewer than 20 mentions or an interval wider than
   6x). "We looked and nothing moved" and "we can't tell yet" are different answers, and
   a PM should hear which one it is.
5. **Side effects.** Every other theme is tested too (FDR-corrected); themes that jump by
   at least 1.5x are reported as suspected regressions.

This is observational evidence. Seasonality, marketing campaigns or a second change
shipped the same week are not controlled for, and the UI says so.

## 10. Evaluation protocol

- **Theme discovery:** adjusted Rand index, NMI, homogeneity and completeness of the
  primary theme against the true topic. Items whose content is all boilerplate have no
  theme and count as their own label. *Recovered* is the share of items whose discovered
  theme is mostly about their true topic (a many-to-one view, which does not penalize a
  topic that is split into precise sub-themes).
- **Release matching:** a release counts as correct if its linked theme is mostly about
  the topic the release was meant to change.
- **Early-warning backtest:** every day from week 6 onward, the trend statuses are
  recomputed with data up to that day only. A spike counts as detected on the first day
  any theme mostly about that topic is *emerging* or *new*. An alert is false if the
  topic had no spike or gradual growth active at the time; consecutive days count as one
  episode. The baseline is the dashboard rule "last 7 days >= 2x the weekly average of
  the previous 4 weeks, and at least 5 mentions".
- **Caveat:** the backtest replays the statistics, not the clustering. Themes are
  learned on the full history, so the backtest is slightly optimistic.

## 11. What did not work

- **TF-IDF + k-means on whole tickets** (the first attempt): ARI 0.12-0.35. Clusters
  followed phrasing and ticket templates.
- **MiniLM + k-means on whole tickets:** ARI 0.39-0.55. One "junk" cluster held 440
  items from all twelve topics; the product name "Tempo" and team-size sentences pulled
  them together.
- **Silhouette-based choice of k:** the score kept increasing with k and favored
  solutions that split the largest themes.
- **Transitive (union-find) consolidation:** merged export requests into praise through
  a chain of pairwise-similar themes. Replaced by complete linkage.
- **Summing the MRR of every account that mentioned a theme:** saturated at the level of
  a few enterprise accounts for every theme, so revenue stopped discriminating.
- **Normalizing trends by total volume:** false "declining" flags for unrelated themes
  during the sync incident (see section 8).
