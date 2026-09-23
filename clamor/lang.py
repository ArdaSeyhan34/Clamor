"""Language packs: everything in Clamor that depends on the language of the feedback.

The statistics are language-agnostic; what changes per language is how text is cleaned
(greetings, sign-offs, forum tags), which sentences count as boilerplate, which words are
stop words, how sentiment is scored and which cue phrases mark a bug or a request.

Turkish needs two extra considerations:

* **Case folding.** Python's ``"İ".lower()`` gives ``"i̇"`` (two code points) and
  ``"I".lower()`` gives ``"i"`` instead of ``"ı"``. :func:`tr_lower` fixes both.
* **Informal spelling.** Reviews are often typed without Turkish characters
  ("calismiyor" for "çalışmıyor"). Lexicons and cue patterns are therefore matched on an
  ASCII-folded form (:func:`fold`), so both spellings hit the same entry.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_TR_FOLD = str.maketrans("çğıöşüâîû", "cgiosuaiu")


def tr_lower(text: str) -> str:
    return text.replace("I", "ı").replace("İ", "i").lower()


def fold(text: str) -> str:
    """Lower-case and strip Turkish diacritics: 'Çalışmıyor' -> 'calismiyor'."""
    return tr_lower(text).translate(_TR_FOLD)


@dataclass(frozen=True)
class LanguagePack:
    code: str
    name: str
    default_embedding: str
    boilerplate_prototypes: tuple[str, ...]
    salutation: re.Pattern  # a whole segment that is only a greeting or a sign-off
    leading_junk: re.Pattern  # tags, greetings and discourse markers at segment start
    stop_words: frozenset[str]
    lexicon: dict[str, float]  # sentiment; keys are matched on normalized tokens
    prefix_lexicon: bool  # Turkish is agglutinative: match lexicon keys as word prefixes
    negations_before: frozenset[str]  # "not great"
    negations_after: frozenset[str]  # "güzel değil"
    intensifiers: dict[str, float]
    kind_cues: dict[str, tuple[str, ...]]
    normalize: Callable[[str], str] = field(default=str.lower)  # lexicon and cue matching
    lower: Callable[[str], str] = field(default=str.lower)  # keyword extraction

    def tokens(self, text: str) -> list[str]:
        return re.findall(r"[^\W\d_]+", self.normalize(text))


# --------------------------------------------------------------------------- English
EN_STOP = frozenset(ENGLISH_STOP_WORDS)

ENGLISH = LanguagePack(
    code="en",
    name="English",
    default_embedding="minilm",
    boilerplate_prototypes=(
        "Hi team",
        "Hello support",
        "Thanks in advance",
        "Thank you!",
        "Best regards, Maria",
        "Please fix this asap.",
        "This is really frustrating.",
        "Very annoying.",
        "Hope this gets fixed soon.",
        "Not happy about this.",
        "Would really appreciate it.",
        "Is this on the roadmap?",
        "Keep up the great work!",
        "Five stars.",
        "We use the product every day.",
        "Our team of 20 people relies on it.",
        "I have been a customer for 2 years.",
        "It is central to how we plan our week.",
        "Renewal is in 4 weeks.",
        "Expansion to 30 more seats depends on this.",
        "They mentioned evaluating competitors.",
        "Otherwise very happy with the product.",
        "Anyone else seeing this?",
        "Any update on this?",
        "Let me know if you need more details.",
    ),
    salutation=re.compile(
        r"^(?:[-–—~]+|(?i:thanks|thank you|thx|cheers|best|regards|kind regards|best regards"
        r"|sincerely|hi|hello|hey|dear)\b)[\s,.!]*(?:[A-Z][\w.'-]*\s*){0,3}[.!]?$"
    ),
    leading_junk=re.compile(
        r"^(\[[^\]]{1,25}\]\s*"
        r"|[+\w][\w +'-]{1,24}:\s+"
        r"|(hi|hello|hey|dear|good (morning|afternoon))\b[^,.!?]{0,25}[,!.]\s*"
        r"|(also|and|but|plus|btw|ps|fyi|anyway|small thing,? but|one more thing)[,:]?\s+)",
        re.I,
    ),
    stop_words=EN_STOP,
    lexicon={
        # strongly negative
        "unusable": -3.0,
        "broken": -2.5,
        "crash": -2.5,
        "crashes": -2.5,
        "crashing": -2.5,
        "terrible": -3.0,
        "awful": -3.0,
        "horrible": -3.0,
        "worst": -3.0,
        "useless": -2.8,
        "unacceptable": -2.8,
        "furious": -3.0,
        "hate": -2.8,
        "disappointed": -2.2,
        "frustrating": -2.2,
        "frustrated": -2.2,
        "unfair": -2.0,
        "unreliable": -2.3,
        "fails": -2.0,
        "failed": -2.0,
        "failing": -2.0,
        "error": -1.8,
        "errors": -1.8,
        "bug": -1.8,
        "buggy": -2.2,
        "freezes": -2.2,
        "freeze": -2.0,
        "hangs": -1.8,
        "missed": -1.6,
        "lost": -1.8,
        "disappear": -2.0,
        "disappeared": -2.0,
        "disappears": -2.0,
        "wrong": -1.6,
        "stuck": -1.6,
        "spam": -2.0,
        "drowning": -2.0,
        "painful": -1.8,
        "confusing": -1.6,
        "confused": -1.4,
        "annoying": -1.8,
        "annoyed": -1.8,
        "slow": -1.5,
        "laggy": -1.8,
        "lag": -1.5,
        "drains": -1.6,
        "duplicate": -1.4,
        "duplicates": -1.4,
        "twice": -0.8,
        "expensive": -1.6,
        "pricey": -1.4,
        "overpriced": -2.2,
        "switching": -1.0,
        "risk": -1.0,
        "blocker": -1.5,
        "problem": -1.2,
        "problems": -1.2,
        "issue": -1.0,
        "issues": -1.0,
        "hurts": -1.5,
        "cannot": -0.8,
        "unhappy": -2.0,
        "forever": -0.8,
        "ignore": -1.0,
        "resets": -1.0,
        "lose": -1.5,
        # positive
        "love": 2.6,
        "loving": 2.4,
        "great": 2.2,
        "excellent": 2.8,
        "amazing": 2.8,
        "fantastic": 2.8,
        "awesome": 2.6,
        "best": 2.4,
        "perfect": 2.6,
        "intuitive": 2.0,
        "easy": 1.6,
        "reliable": 1.8,
        "fast": 1.4,
        "clean": 1.2,
        "responsive": 1.6,
        "happy": 1.8,
        "helpful": 1.8,
        "nice": 1.4,
        "good": 1.4,
        "saves": 1.6,
        "worth": 1.4,
        "appreciate": 1.2,
        "thanks": 0.6,
        "thank": 0.6,
        "smooth": 1.4,
        "simple": 1.0,
        "recommend": 1.8,
        "works": 0.6,
        "improved": 1.2,
        "better": 1.0,
        "fixed": 0.8,
    },
    prefix_lexicon=False,
    negations_before=frozenset(
        {
            "not",
            "no",
            "never",
            "dont",
            "doesnt",
            "didnt",
            "isnt",
            "wasnt",
            "cant",
            "cannot",
            "wont",
            "without",
            "hardly",
        }
    ),
    negations_after=frozenset(),
    intensifiers={
        "very": 1.3,
        "really": 1.3,
        "so": 1.2,
        "extremely": 1.5,
        "completely": 1.4,
        "totally": 1.4,
        "super": 1.3,
        "incredibly": 1.5,
        "way": 1.2,
    },
    kind_cues={
        "bug": (
            r"\bcrash",
            r"\bbroken\b",
            r"\berror",
            r"\bfail",
            r"\bbug",
            r"\bfreez",
            r"\bhang",
            r"\bstuck\b",
            r"\bdisappear",
            r"\bduplicate",
            r"\bnot working\b",
            r"\bwrong\b",
            r"\bslow\b",
            r"\blag",
            r"\bstopped\b",
            r"\bunreliable\b",
            r"\bnever (make|sync|load)",
            r"\btwice\b",
            r"\bdrains?\b",
            r"\bunusable\b",
            r"\bforever\b",
            r"\bsync\w* (fail|problem)",
            r"\bis broken\b",
            r"\bno longer\b",
        ),
        "feature_request": (
            r"\bplease add\b",
            r"\badd an?\b",
            r"\bwould love\b",
            r"\bwish\b",
            r"\bneed\b",
            r"\bsupport for\b",
            r"\bwould be (great|amazing|nice)\b",
            r"\bintegration\b",
            r"\bfeature\b",
            r"\boption\b",
            r"\blet me\b",
            r"\bcan (you|we)\b",
            r"\bany plans\b",
            r"\broadmap\b",
            r"\brequire",
            r"\bexport\b",
            r"\bplease\b",
        ),
        "pricing": (
            r"\bprice",
            r"\bpricing\b",
            r"\bexpensive\b",
            r"\bcost",
            r"\bpay",
            r"\bper[- ]seat\b",
            r"\bbilling\b",
            r"\bbilled\b",
            r"\bcheaper\b",
            r"\bpricey\b",
            r"\bplan costs?\b",
        ),
    },
)

# --------------------------------------------------------------------------- Turkish
_TR_STOP_WORDS = """
acaba ama ancak artık aslında az bana bazı belki ben beni benim bile bir biraz birçok biri
birkaç birşey biz bize bizi bizim bu buna bunda bundan bunu bunun burada bütün çok çünkü da
daha de defa değil diye dolayı en eğer gibi hala hatta hem hemen hep hepsi her herhangi
herkes hiç için ile ilgili ise işte kadar kendi kez ki kim mı mi mu mü nasıl ne neden nerede
nereye niçin niye o olan olarak oldu olduğu olsa olsun olup olur olursa on ona ondan onlar
onu onun orada öyle pek rağmen sadece sanki sen siz size sizin şey şeyi şekilde şimdi şu
şuna şunu tabi tam tüm üzere ve veya ya yani yine yok zaten zira lütfen gerçekten cidden
bile hâlâ yapıyorum yaptım yapmak etmek ediyorum ettim oluyor olmuyor olması var
uygulama uygulamada uygulamayı uygulamanın uygulaması app
"""
_TR_STOP = frozenset(_TR_STOP_WORDS.split())

TURKISH = LanguagePack(
    code="tr",
    name="Türkçe",
    default_embedding="hybrid",  # multilingual MiniLM + TF-IDF, see docs/methodology.md
    boilerplate_prototypes=(
        "Merhaba",
        "Merhabalar, iyi günler",
        "İyi çalışmalar",
        "Kolay gelsin",
        "Teşekkürler",
        "Teşekkür ederim, iyi çalışmalar",
        "Saygılarımla, Ayşe",
        "Lütfen acilen çözün.",
        "Çok sinir bozucu.",
        "Dönüşünüzü bekliyorum.",
        "Yardımcı olursanız sevinirim.",
        "Mağdur durumdayım.",
        "Değerlendirirseniz sevinirim.",
        "Yol haritanızda var mı?",
        "Her gün öğle yemeğinde kullanıyorum.",
        "Şirketimiz geçen ay bu karta geçti.",
        "Yaklaşık iki yıldır kullanıcınızım.",
        "Emeği geçenlere teşekkürler!",
        "Başarılar!",
        "5 yıldızı hak ediyor.",
        "Başka bir sorun yok.",
        "Çözüm bekliyorum.",
        "Rezalet.",
        "Berbat.",
        "Telefon numaram [phone], beni arayabilirsiniz.",
        "E-posta adresim [email].",
        "Bu konuda bilgi verir misiniz?",
    ),
    salutation=re.compile(
        r"^(?:[-–—~]+|(?i:merhaba(?:lar)?|selam(?:lar)?|iyi (?:günler|gunler|çalışmalar"
        r"|calismalar|akşamlar|aksamlar)|kolay gelsin|teşekkür(?:ler| ederim)"
        r"|tesekkur(?:ler| ederim)|saygılarımla|saygilarimla|sevgiler|hayırlı işler)\b)"
        r"[\s,.!]*(?:[A-ZÇĞİÖŞÜ][\wçğıöşü.'-]*\s*){0,3}[.!]?$"
    ),
    leading_junk=re.compile(
        r"^(\[[^\]]{1,25}\]\s*"
        r"|[+\wçğıöşüÇĞİÖŞÜ][\wçğıöşüÇĞİÖŞÜ +'-]{1,24}:\s+"
        r"|(merhaba(lar)?|selam(lar)?|iyi günler|iyi gunler|iyi çalışmalar|iyi calismalar)"
        r"\b[^,.!?]{0,25}[,!.]\s*"
        r"|(ayrıca|ayrica|bir de|bu arada|ek olarak|son olarak|bir şey daha|bir sey daha)"
        r"[,:]?\s+)",
        re.I,
    ),
    stop_words=_TR_STOP | frozenset(fold(w) for w in _TR_STOP) | EN_STOP,
    lexicon={
        # negative (ASCII-folded stems, matched as word prefixes)
        "rezalet": -3.0,
        "berbat": -3.0,
        "felaket": -3.0,
        "facia": -3.0,
        "skandal": -2.5,
        "sacma": -2.0,
        "calismiyor": -2.5,
        "calismadi": -2.5,
        "calismaz": -2.3,
        "acilmiyor": -2.2,
        "acilmadi": -2.2,
        "okutmuyor": -2.2,
        "okumuyor": -2.2,
        "gecmiyor": -2.0,
        "gecmedi": -2.0,
        "yuklenmiyor": -2.0,
        "yuklenmedi": -2.0,
        "baglanmiyor": -2.0,
        "gelmiyor": -1.6,
        "gelmedi": -1.6,
        "donuyor": -2.0,
        "donmus": -2.0,
        "cokuyor": -2.5,
        "coktu": -2.5,
        "kapaniyor": -2.0,
        "takili": -1.8,
        "hata": -1.8,
        "sorun": -1.3,
        "problem": -1.3,
        "yavas": -1.5,
        "kotu": -2.0,
        "sinir": -2.0,
        "magdur": -2.2,
        "ulasilamiyor": -2.0,
        "ulasamiyorum": -2.0,
        "maalesef": -1.2,
        "sikayet": -1.5,
        "yetersiz": -1.8,
        "eksik": -1.0,
        "karmasik": -1.3,
        "anlamsiz": -1.5,
        "pisman": -2.0,
        "haksiz": -2.0,
        "imkansiz": -2.0,
        "bekletiyor": -1.8,
        "bekliyorum": -1.0,
        "rezil": -2.5,
        "yanlis": -1.4,
        "giremiyorum": -2.2,
        "yapamiyorum": -2.0,
        "edemiyorum": -1.8,
        "yanmasi": -1.2,
        "yaniyor": -1.2,
        "cevap vermiyor": -1.8,
        "yansimadi": -1.8,
        "dusmuyor": -1.8,
        "dusmedi": -1.8,
        "tuketiyor": -1.0,
        "yoruyor": -1.2,
        "ne yazik": -1.5,
        # positive
        "harika": 2.6,
        "mukemmel": 2.8,
        "super": 2.2,
        "guzel": 1.8,
        "iyi": 1.4,
        "pratik": 1.8,
        "kolay": 1.6,
        "hizli": 1.4,
        "tesekkur": 0.6,
        "memnun": 2.0,
        "basarili": 2.0,
        "sorunsuz": 2.0,
        "kullanisli": 1.8,
        "begendim": 2.0,
        "seviyorum": 2.4,
        "tavsiye": 1.8,
        "efsane": 2.4,
        "muthis": 2.6,
        "rahat": 1.4,
        "sade": 1.0,
        "anlasilir": 1.4,
        "kurtuldum": 1.4,
        "basit": 1.0,
        "en iyi": 2.4,
        # longer forms that would otherwise be caught by a shorter stem of opposite sign
        "hatasiz": 1.5,
        "rahatsiz": -1.5,
        "eksiksiz": 1.2,
        # greetings and sign-offs carry no opinion
        "iyi gunler": 0.0,
        "iyi calismalar": 0.0,
        "iyi aksamlar": 0.0,
        "kolay gelsin": 0.0,
    },
    prefix_lexicon=True,
    negations_before=frozenset(),
    negations_after=frozenset({"degil", "degildi"}),
    intensifiers={
        "cok": 1.3,
        "asiri": 1.5,
        "gercekten": 1.3,
        "cidden": 1.3,
        "son": 1.1,
        "resmen": 1.3,
        "fazla": 1.2,
        "bayagi": 1.2,
        "hic": 1.3,
    },
    kind_cues={
        "bug": (
            r"calism(iyor|adi|az)",
            r"acilmi",
            r"okutmu",
            r"okumu",
            r"gecmi(yor|di)",
            r"yuklenm(edi|iyor)",
            r"baglanmi",
            r"gelmi(yor|di)",
            r"donuyor|donmus|donma",
            r"cok(uyor|tu)",
            r"kapaniyor",
            r"\bhata",
            r"\bsorun(?!suz)",
            r"\bproblem",
            r"yavas",
            r"takil",
            r"\bbug",
            r"dus(med|mu)",
            r"gorunmu",
            r"yanlis",
            r"giremiyorum",
            r"yansimadi",
            r"iki (kez|kere)",
            r"zaman asimi",
            r"cekildi",
        ),
        "feature_request": (
            r"eklen(meli|se|ir mi|mesi|irse|sin)",
            r"\bolsa\b",
            r"olsa (cok|harika|iyi)",
            r"keske",
            r"isti(yoruz|yorum)|isterdim|isterim",
            r"ozellik",
            r"secenek",
            r"olmali",
            r"lutfen ekle",
            r"entegrasyon",
            r"destekle",
            r"kullanilabilse",
            r"gecerli olsa",
            r"eklenebilse",
            r"harika olur",
        ),
        "pricing": (
            r"ucret",
            r"komisyon",
            r"kesinti",
            r"aidat",
            r"pahali",
            r"fiyat",
        ),
    },
    normalize=fold,
    lower=tr_lower,
)

LANGUAGES: dict[str, LanguagePack] = {"en": ENGLISH, "tr": TURKISH}


def get_language(code: str) -> LanguagePack:
    try:
        return LANGUAGES[code]
    except KeyError:
        raise ValueError(f"Unsupported language {code!r}; use one of {sorted(LANGUAGES)}") from None


def with_ascii_variants(prototypes: tuple[str, ...]) -> tuple[str, ...]:
    """Add each prototype typed without Turkish characters ("Mağdur" -> "Magdur")."""
    table = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
    extra = tuple(p.translate(table) for p in prototypes if p.translate(table) != p)
    return prototypes + extra
