"""Second demo scenario: **Lezzo**, a fictional Turkish employee meal-card app.

Employers load a monthly meal allowance; employees pay at partner restaurants with a QR
code or NFC and track their balance in the app. The scenario mirrors a common real-world
setup: **only end-user feedback** (app-store reviews, in-app support tickets and an in-app
survey), written in Turkish, with **no revenue data**. So Clamor has to rank by reach,
severity and momentum alone, and cope with informal Turkish (missing diacritics, typos,
lower-case reviews).

What the simulation plants:

* a loud, stable request: more partner restaurants (improved by v5.0),
* a login regression introduced by v5.1 and fixed by v5.1.1,
* a performance release (v5.2) that changes nothing,
* a new QR payment screen (v5.3) that breaks payments while the data ends,
* balance-loading complaints that spike at the start of every month (payday),
* personal data (phone numbers, e-mail addresses) in support tickets, to exercise masking.

Names, companies, phone numbers (``0500 000 ...``) and e-mail addresses
(``@example.com``) are fictional or reserved and cannot belong to real people.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .synth import (
    DEFAULT_DAYS,
    START_DATE,
    Event,
    Release,
    SyntheticDataset,
    ThemeSpec,
    _fill,
    _pick,
    _rate_multiplier,
    _typo,
)

PRODUCT = "Lezzo"
_ASCII = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")

SLOTS: dict[str, list[str]] = {
    "since": ["yeni güncellemeden beri", "iki gündür", "bu hafta", "sabahtan beri",
              "son güncellemeden sonra", "dünden beri"],
    "device": ["Samsung", "Xiaomi", "iPhone", "eski Android", "Huawei", "Oppo"],
    "district": ["Kadıköy", "Ataşehir", "Maslak", "Levent", "Çankaya", "Bornova", "Nilüfer",
                 "Konak"],
    "amount": ["50", "120", "250", "400"],
    "months": ["3", "6", "8", "12", "18"],
    "name": ["Ayşe", "Mehmet", "Zeynep", "Emre", "Elif", "Burak", "Selin", "Can", "Deniz",
             "Ece", "Mert", "Ceren"],
    "phone": ["0500 000 12 34", "0500 000 45 67", "0500 000 89 01"],
    "email": ["ayse.k@example.com", "mehmet.y@example.com", "zeynep.a@example.com"],
}  # fmt: skip

ALL_USERS = {"user": 1.0}

THEMES: tuple[ThemeSpec, ...] = (
    ThemeSpec(
        "qr_payment", "bug", "negative", 9.0, ALL_USERS,
        (
            "QR kodu okutunca ödeme ekranında takılı kalıyor",
            "kasada QR okutuyorum ama ödeme bir türlü geçmiyor",
            "QR ile ödeme {since} çalışmıyor, kasada rezil oldum",
            "QR okuttuktan sonra uygulama beyaz ekranda kalıyor",
            "ödeme onaylandı diyor ama restoranın cihazına düşmüyor",
            "kamera QR kodu okumuyor, sürekli odaklanmaya çalışıyor",
            "QR ödemesi iki kere çekildi, bakiyemden iki kez düştü",
        ),
        {"app_store": 1.2},
    ),
    ThemeSpec(
        "nfc_payment", "bug", "negative", 4.0, ALL_USERS,
        (
            "NFC ile ödeme yapmak istiyorum ama telefonu yaklaştırınca hiçbir şey olmuyor",
            "temassız ödeme {device} telefonumda çalışmıyor",
            "NFC ödemesi sürekli zaman aşımına uğruyor",
            "telefonu POS cihazına yaklaştırıyorum, ödeme ekranı açılmıyor",
            "temassız ödeme bazen geçiyor bazen geçmiyor, güvenemiyorum",
        ),
    ),
    ThemeSpec(
        "balance_loading", "bug", "negative", 6.0, ALL_USERS,
        (
            "maaş günü geldi ama yemek bakiyem hala yüklenmedi",
            "bu ay bakiyem geç yüklendi, iki gün yemek kartsız kaldım",
            "şirketim yükleme yaptığını söylüyor ama bakiyem sıfır görünüyor",
            "bakiye yüklendi bildirimi geldi ama uygulamada eski bakiye duruyor",
            "yüklenen tutar eksik görünüyor, {amount} TL eksik",
        ),
        {"support_ticket": 1.6},
    ),
    ThemeSpec(
        "restaurant_network", "feature_request", "neutral", 14.0, ALL_USERS,
        (
            "anlaşmalı restoran sayısı çok az, {district} tarafında neredeyse hiç yer yok",
            "evimin ve iş yerimin yakınında kartın geçtiği restoran yok",
            "sevdiğim kafeler anlaşmalı değil, lütfen daha fazla restoran ekleyin",
            "market alışverişinde de kullanılabilse çok iyi olur",
            "kahve zincirlerinde de geçerli olsa harika olur",
            "yemek siparişi uygulamalarında da kullanılabilse harika olur",
            "restoran ağı genişlerse kartı çok daha fazla kullanırım",
        ),
        {"app_store": 1.3, "survey": 1.4},
    ),
    ThemeSpec(
        "login", "bug", "negative", 3.0, ALL_USERS,
        (
            "SMS doğrulama kodu gelmiyor, uygulamaya giremiyorum",
            "giriş yaparken sürekli oturum süreniz doldu hatası alıyorum",
            "şifremi doğru girmeme rağmen giriş yapamıyorum",
            "doğrulama kodu çok geç geliyor, geldiğinde de süresi dolmuş oluyor",
            "yeni giriş ekranından sonra hesabıma erişemiyorum",
            "her açılışta tekrar tekrar SMS kodu istiyor",
        ),
        {"support_ticket": 1.4},
    ),
    ThemeSpec(
        "app_performance", "bug", "negative", 9.0, ALL_USERS,
        (
            "uygulama çok yavaş açılıyor, kasada sırayı bekletiyorum",
            "uygulama sürekli donuyor, kapatıp açmam gerekiyor",
            "{device} telefonumda açılışta çöküyor",
            "ana sayfanın yüklenmesi yarım dakika sürüyor",
            "uygulama çok fazla pil tüketiyor",
            "bakiye ekranı açılana kadar kasadaki herkes bekliyor",
        ),
        {"app_store": 1.5},
    ),
    ThemeSpec(
        "map_search", "ux", "negative", 5.0, ALL_USERS,
        (
            "haritada restoranlar yanlış konumda gösteriliyor",
            "restoran ararken filtreler çalışmıyor",
            "yakınımdaki restoranları listelemiyor, konumumu bulamıyor",
            "haritada kapanmış restoranlar hala görünüyor",
            "restoranların çalışma saatleri yanlış yazıyor",
        ),
    ),
    ThemeSpec(
        "dark_mode", "feature_request", "neutral", 5.0, ALL_USERS,
        (
            "karanlık mod eklenirse çok iyi olur",
            "gece kullanırken beyaz ekran göz yoruyor, karanlık tema lütfen",
            "karanlık mod seçeneği yok mu?",
            "sistem temasına uyan bir koyu tema eklenmeli",
        ),
        {"app_store": 1.4},
    ),
    ThemeSpec(
        "wallet", "feature_request", "neutral", 6.0, ALL_USERS,
        (
            "Apple Pay'e eklenebilse telefonu açmadan ödeme yapardım",
            "Google Pay entegrasyonu olsa çok pratik olur",
            "kartı telefonun cüzdan uygulamasına eklemek istiyorum",
            "akıllı saatle ödeme yapabilmek harika olurdu",
        ),
    ),
    ThemeSpec(
        "balance_expiry", "pricing", "negative", 5.0, ALL_USERS,
        (
            "ay sonunda kalan bakiye yanıyor mu, bir sonraki aya devretmesi lazım",
            "kullanmadığım bakiye sonraki aya aktarılmıyor, bu çok haksız",
            "bakiyenin son kullanma tarihi olduğunu kimse söylemedi",
            "kalan bakiyemi bir iş arkadaşıma transfer edemiyorum",
        ),
        {"survey": 1.5},
    ),
    ThemeSpec(
        "customer_support", "ux", "negative", 5.0, ALL_USERS,
        (
            "müşteri hizmetlerine ulaşmak imkansız, çağrı merkezi yarım saat bekletiyor",
            "canlı destek hiç cevap vermiyor",
            "şikayet yazdım, üç gündür dönüş yapılmadı",
            "uygulamada destek talebi oluşturma seçeneği bulamadım",
        ),
        {"support_ticket": 1.4},
    ),
    ThemeSpec(
        "refund", "bug", "negative", 3.0, ALL_USERS,
        (
            "iptal edilen siparişin iadesi hala bakiyeme yansımadı",
            "restoran ödemeyi iptal etti ama para geri gelmedi",
            "iade süreci çok uzun, bir haftadır bekliyorum",
        ),
        {"support_ticket": 1.8},
    ),
    ThemeSpec(
        "praise", "praise", "positive", 10.0, ALL_USERS,
        (
            "kart taşımaktan kurtuldum, harika bir uygulama",
            "bakiye takibi çok kolay, teşekkürler",
            "kullanımı çok basit ve hızlı",
            "yemek kartı uygulamaları arasında en iyisi",
            "arayüz sade ve anlaşılır",
            "ödeme saniyeler içinde bitiyor, çok memnunum",
        ),
        {"support_ticket": 0.3, "survey": 1.3},
    ),
)  # fmt: skip

RELEASES: tuple[Release, ...] = (
    Release(30, "v5.0", "Restoran ağı genişlemesi",
            "Anlaşmalı restoran ağına 1.500 yeni restoran ve kafe eklendi; kartınızı artık çok "
            "daha fazla noktada kullanabilirsiniz.", "restaurant_network"),
    Release(60, "v5.1", "Yeni giriş deneyimi",
            "SMS doğrulamalı yeni giriş ekranı ve daha güvenli oturum yönetimi.", "login"),
    Release(88, "v5.1.1", "Giriş düzeltmesi",
            "SMS doğrulama kodlarının gecikmesi ve oturum süresi hataları giderildi.", "login"),
    Release(120, "v5.2", "Performans iyileştirmeleri",
            "Uygulamanın açılış süresi kısaltıldı, donma ve yavaşlık sorunları giderildi.",
            "app_performance"),
    Release(158, "v5.3", "Yeni QR ödeme ekranı",
            "QR ile ödeme ekranı yeniden tasarlandı, kasada ödeme artık daha hızlı.",
            "qr_payment"),
)  # fmt: skip

EVENTS: tuple[Event, ...] = (
    Event("restaurant_network", 31, 0.55, ramp_days=14),
    Event("login", 61, 7.0, end_day=89, ramp_days=2),
    Event("app_performance", 121, 1.0),  # the release that changed nothing
    Event("qr_payment", 159, 5.0, ramp_days=3),
)

CHANNEL_MIX = {"app_store": 0.50, "support_ticket": 0.32, "survey": 0.18}

CLOSERS = {
    "negative": ["Lütfen acilen çözün.", "Çok sinir bozucu.", "Dönüşünüzü bekliyorum.",
                 "Mağdur durumdayım.", "Rezalet.", "", ""],
    "neutral": ["Teşekkürler.", "Değerlendirirseniz sevinirim.", "Yol haritanızda var mı?",
                "", ""],
    "positive": ["Emeği geçenlere teşekkürler!", "Başarılar!", "5 yıldızı hak ediyor.", "", ""],
}  # fmt: skip
CONTEXT = [
    "Her gün öğle yemeğinde kullanıyorum.",
    "Şirketimiz geçen ay bu karta geçti.",
    "Yaklaşık {months} aydır kullanıcınızım.",
    "Telefon numaram {phone}, beni arayabilirsiniz.",
    "E-posta adresim {email}.",
    "",
    "",
    "",
]
GREETINGS = ["Merhaba,", "Merhabalar,", "İyi günler,", "Selam,", ""]
SIGNOFFS = ["Teşekkürler, {name}", "İyi çalışmalar, {name}", "Saygılarımla, {name}", "- {name}", ""]


def _sentence(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    if not text[1:2].isupper():
        text = ("İ" if text[0] == "i" else text[0].upper()) + text[1:]
    return text if text[-1] in ".!?" else text + "."


def _render(theme: ThemeSpec, channel: str, rng: np.random.Generator) -> tuple[str, float | None]:
    phrase = _fill(_pick(list(theme.phrases), rng), rng, SLOTS)
    extra = ""
    if rng.random() < 0.12:
        other = THEMES[rng.integers(len(THEMES))]
        if other.key not in {theme.key, "praise"}:
            extra = " Ayrıca " + _fill(_pick(list(other.phrases), rng), rng, SLOTS) + "."
    closer = _pick(CLOSERS[theme.polarity], rng)
    rating: float | None = None
    if channel == "app_store":
        stars = {"negative": [1, 1, 1, 2, 2, 3], "neutral": [3, 4, 4, 5], "positive": [4, 5, 5]}
        rating = float(_pick(stars[theme.polarity], rng))
        text = f"{_sentence(phrase)}{extra} {closer}"
    elif channel == "survey":
        scores = {"negative": range(0, 7), "neutral": range(5, 9), "positive": range(8, 11)}
        rating = float(_pick(list(scores[theme.polarity]), rng))
        text = f"{_sentence(phrase)}{extra}"
    else:  # support ticket (in-app form, e-mail or call-center note)
        parts = [
            _pick(GREETINGS, rng),
            _fill(_pick(CONTEXT, rng), rng, SLOTS),
            _sentence(phrase) + extra,
            closer,
            _fill(_pick(SIGNOFFS, rng), rng, SLOTS),
        ]
        text = " ".join(p for p in parts if p)
    if rng.random() < 0.15 and channel == "app_store":  # "Lezzo QR kodu okumuyor..."
        text = f"{PRODUCT} {text[0].lower() if text[1:2].islower() else text[0]}{text[1:]}"
    text = re.sub(r"\s+", " ", text).strip()
    if rng.random() < 0.15:
        text = _typo(text, rng)
    if channel in {"app_store", "survey"} and rng.random() < 0.30:
        text = text.translate(_ASCII)  # typed without Turkish characters
    if channel == "app_store" and rng.random() < 0.25:
        text = text.replace("I", "ı").replace("İ", "i").lower()
    return text, rating


def _payday_factor(theme: ThemeSpec, date: pd.Timestamp) -> float:
    """Balance-loading complaints spike in the first days of every month."""
    return 2.5 if theme.key == "balance_loading" and date.day <= 3 else 1.0


def generate(
    seed: int = 11,
    n_users: int = 2500,
    days: int = DEFAULT_DAYS,
    volume_scale: float = 1.25,
) -> SyntheticDataset:
    """Simulate `days` of Lezzo feedback. Deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    users = [f"U-{i + 1:05d}" for i in range(n_users)]
    # a minority of users writes a lot: heavy-tailed activity
    activity = rng.pareto(2.0, n_users) + 1
    activity /= activity.sum()
    channels = list(CHANNEL_MIX)

    feedback_rows, truth_rows = [], []
    for day in range(days):
        date = START_DATE + pd.Timedelta(days=day)
        weekday_factor = 0.7 if date.weekday() >= 5 else 1.1
        for theme in THEMES:
            lam = (theme.base_rate / 7.0 * _rate_multiplier(theme, day, EVENTS)
                   * weekday_factor * _payday_factor(theme, date))  # fmt: skip
            for _ in range(rng.poisson(lam * volume_scale)):
                user = users[rng.choice(n_users, p=activity)]
                w = np.array([CHANNEL_MIX[c] * theme.channel_boost.get(c, 1.0) for c in channels])
                channel = channels[rng.choice(len(channels), p=w / w.sum())]
                text, rating = _render(theme, channel, rng)
                ts = date + pd.Timedelta(seconds=int(rng.integers(8 * 3600, 23 * 3600)))
                fid = f"LZ-{len(feedback_rows) + 1:05d}"
                feedback_rows.append(
                    {
                        "feedback_id": fid,
                        "created_at": ts,
                        "channel": channel,
                        "account_id": user,
                        "rating": rating,
                        "text": text,
                    }
                )
                truth_rows.append({"feedback_id": fid, "true_theme": theme.key,
                                   "true_polarity": theme.polarity})  # fmt: skip

    feedback = pd.DataFrame(feedback_rows).sort_values("created_at", kind="stable")
    releases = pd.DataFrame([
        {"date": (START_DATE + pd.Timedelta(days=r.day)).date().isoformat(),
         "version": r.version, "title": r.title, "description": r.description}
        for r in RELEASES
    ])  # fmt: skip
    events = pd.DataFrame(
        [
            {
                "theme": e.theme,
                "start_date": (START_DATE + pd.Timedelta(days=e.start_day)).date().isoformat(),
                "end_date": None if e.end_day is None
                else (START_DATE + pd.Timedelta(days=e.end_day)).date().isoformat(),
                "multiplier": e.multiplier,
                "kind": "spike" if e.multiplier > 1 else ("drop" if e.multiplier < 1 else "none"),
            }
            for e in EVENTS
        ]
        + [{"theme": "balance_loading", "start_date": START_DATE.date().isoformat(),
            "end_date": None, "multiplier": float("nan"), "kind": "monthly pattern"}]
    )  # fmt: skip
    return SyntheticDataset(
        feedback=feedback.reset_index(drop=True),
        accounts=None,
        releases=releases,
        ground_truth=pd.DataFrame(truth_rows),
        events=events,
        release_truth=pd.DataFrame(
            [{"version": r.version, "intended_theme": r.theme_hint} for r in RELEASES]
        ),
    )
