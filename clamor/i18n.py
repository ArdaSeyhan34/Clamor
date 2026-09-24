"""Output language of reports, briefs, charts and the dashboard.

The analysis language (``Config.language``) decides how feedback is *read*; this module
decides how results are *written*. Strings stay in English in the code and are looked up
here, gettext style: ``t("Top priority: {name}", "tr", name=...)``. A string without a
translation falls back to English, and ``tests/test_i18n.py`` checks that every literal
passed to :func:`t` has a Turkish entry.
"""

from __future__ import annotations

import pandas as pd

OUTPUT_LANGUAGES = ("en", "tr")

TR_MONTHS = [
    "Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran",
    "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık",
]  # fmt: skip

TR: dict[str, str] = {
    # ---- labels: theme status, type, release verdict, signals, presets, people
    "★ new": "★ yeni",
    "▲ emerging": "▲ hızla artıyor",
    "↗ rising": "↗ artıyor",
    "– stable": "– durağan",
    "↘ declining": "↘ azalıyor",
    "· quiet": "· az veri",
    "new": "yeni",
    "emerging": "hızla artıyor",
    "rising": "artıyor",
    "stable": "durağan",
    "declining": "azalıyor",
    "quiet": "az veri",
    "Bug": "Hata",
    "Feature request": "Özellik isteği",
    "Usability": "Kullanılabilirlik",
    "Pricing": "Fiyatlandırma",
    "Praise": "Övgü",
    "Other": "Diğer",
    "Bug / reliability": "Hata / güvenilirlik",
    "Pricing & packaging": "Fiyatlandırma ve paketleme",
    "Resolved": "Çözüldü",
    "Improved": "İyileşti",
    "Worse": "Kötüleşti",
    "No detectable change": "Belirgin değişiklik yok",
    "Inconclusive": "Belirsiz",
    "Not enough data": "Yeterli veri yok",
    "No matching theme": "Eşleşen tema yok",
    "Reach": "Erişim",
    "Revenue": "Gelir",
    "Severity": "Şiddet",
    "Momentum": "İvme",
    "reach": "erişim",
    "revenue": "gelir",
    "severity": "şiddet",
    "momentum": "ivme",
    "balanced": "dengeli",
    "growth": "büyüme",
    "enterprise": "kurumsal",
    "quality": "kalite",
    "accounts": "hesap",
    "users": "kullanıcı",
    "Accounts": "Hesaplar",
    "Users": "Kullanıcılar",
    "accounts (plural)": "hesaplar",
    "users (plural)": "kullanıcılar",
    "app store": "uygulama mağazası",
    "support ticket": "destek talebi",
    "survey": "anket",
    "nps survey": "NPS anketi",
    "sales call": "satış görüşmesi",
    "community": "topluluk",
    # ---- headline insights
    "Top priority: {name}": "En yüksek öncelik: {name}",
    (
        "Score {score}/100, driven mostly by {a} and {b}: {mentions} mentions from {n} {people} "
        "in the last {window} days."
    ): (
        "Puan {score}/100, en çok {a} ve {b} belirliyor: son {window} günde {n} {people} "
        "tarafından {mentions} kez bahsedildi."
    ),
    "Early warning: {name} is {status}": "Erken uyarı: {name} ({status})",
    ("Mention rate is {lift}x its baseline over the last {days} days (95% CI {lo}-{hi}, q {q})."): (
        "Son {days} günde bahsedilme oranı olağan düzeyinin {lift} katı (%95 GA {lo}-{hi}, q {q})."
    ),
    "Loud, but not the most valuable: {name}": "Çok konuşuluyor ama en değerlisi değil: {name}",
    (" {share} of its mentions come from {plan} accounts."): (
        " Bahsedenlerin {share} kadarı {plan} planındaki hesaplar."
    ),
    ("Frequently mentioned, but lower priority: {name}"): (
        "Sık bahsediliyor ama önceliği daha düşük: {name}"
    ),
    "fewer distinct users": "daha az sayıda farklı kullanıcı",
    "milder sentiment": "daha ılımlı duygu",
    "no significant growth": "anlamlı bir artış olmaması",
    " It ranks lower mainly because of {reason}.": " Daha aşağıda olmasının asıl nedeni {reason}.",
    ("#{votes} by raw mention count, #{rank} by priority.{extra}"): (
        "Bahsedilme sayısında #{votes}, öncelikte #{rank}.{extra}"
    ),
    "Quiet, but expensive: {name}": "Sessiz ama pahalı: {name}",
    (
        "Only #{votes} by mention count, yet the largest revenue-weighted demand ({mrr} MRR; "
        "accounts raising it hold {arr} ARR). {share} of mentions come from {plan}."
    ): (
        "Bahsedilme sayısında yalnızca #{votes}, ama gelir ağırlıklı talebi en yüksek konu ({mrr} "
        "MRR; dile getiren hesapların toplamı {arr} ARR). Bahsedenlerin {share} kadarı {plan} "
        "planında."
    ),
    (
        "Mentions of “{theme}” changed {change} relative to overall feedback after the release "
        "(95% CI x{lo}-x{hi})."
    ): (
        "Sürümden sonra “{theme}” konusunun tüm geri bildirim içindeki payı {change} (%95 "
        "GA x{lo}-x{hi})."
    ),
    # ---- report tables
    "Rank": "Sıra",
    "Theme": "Tema",
    "Type": "Tür",
    "Score": "Puan",
    "Mentions": "Bahsedilme",
    "Trend": "Eğilim",
    "Rank by mentions": "Bahsedilme sırası",
    "Revenue-weighted MRR": "Gelir ağırlıklı MRR",
    "Date": "Tarih",
    "Release": "Sürüm",
    "Linked theme": "İlgili tema",
    "Before → after": "Önce → sonra",
    "Rate change": "Oran değişimi",
    "Verdict": "Sonuç",
    # ---- Markdown report
    "Clamor report": "Clamor raporu",
    (
        "*{items} feedback items from {n} {people}, {start} to {end}. {themes} themes discovered "
        "with the `{backend}` embedding backend. Priority reflects the last {window} days.*"
    ): (
        "*{start} - {end} arasında {n} {people} tarafından yazılmış {items} geri bildirim. "
        "`{backend}` gömme modeliyle {themes} tema bulundu. Öncelik son {window} günü yansıtır.*"
    ),
    "Key insights": "Öne çıkanlar",
    "Prioritized roadmap": "Önceliklendirilmiş yol haritası",
    "Early warnings": "Erken uyarılar",
    ("No theme is growing significantly faster than feedback overall."): (
        "Hiçbir tema, geri bildirimin geneline göre anlamlı biçimde hızlı büyümüyor."
    ),
    "Status": "Durum",
    "Rate vs baseline": "Olağan düzeye oran",
    "95% CI": "%95 GA",
    "q-value": "q değeri",
    "Release radar": "Sürüm radarı",
    (
        "Did each release change what customers talk about? Rates are compared in windows of up "
        "to {days} days before and after each release, normalized for overall feedback volume."
    ): (
        "Her sürüm, müşterilerin konuştuklarını değiştirdi mi? Oranlar her sürümden önceki ve "
        "sonraki en fazla {days} günlük pencerelerde, toplam geri bildirim hacmine göre "
        "normalleştirilerek karşılaştırılır."
    ),
    ("**Suspected side effects** (themes that spiked after a release they were not linked to):"): (
        "**Olası yan etkiler** (bağlı olmadıkları bir sürümden sonra sıçrayan temalar):"
    ),
    ("- {version}: *{theme}* x{ratio} ({pre} → {post} mentions, q = {q})"): (
        "- {version}: *{theme}* x{ratio} ({pre} → {post} bahsedilme, q = {q})"
    ),
    ("Accuracy against ground truth (synthetic data)"): (
        "Gerçek etiketlere göre doğruluk (sentetik veri)"
    ),
    "Metric": "Ölçüt",
    "Value": "Değer",
    "Adjusted Rand index": "Düzeltilmiş Rand indeksi",
    "Normalized mutual information": "Normalleştirilmiş karşılıklı bilgi",
    "Homogeneity (theme purity)": "Homojenlik (tema saflığı)",
    ("Items landing in a theme about their true topic"): (
        "Gerçek konusuyla ilgili bir temaya düşen öğeler"
    ),
    "Sentiment sign accuracy": "Duygu yönü doğruluğu",
    "Releases linked to the right theme": "Doğru temaya bağlanan sürümler",
    "Spikes detected (Clamor / naive 2x rule)": "Yakalanan sıçramalar (Clamor / basit 2x kuralı)",
    "Median days to detect (Clamor / naive)": "Yakalama süresi, medyan gün (Clamor / basit)",
    "False alarm episodes (Clamor / naive)": "Yanlış alarm sayısı (Clamor / basit)",
    ("*Generated by [Clamor](https://github.com/ArdaSeyhan34/clamor).*"): (
        "*[Clamor](https://github.com/ArdaSeyhan34/clamor) ile oluşturuldu.*"
    ),
    # ---- HTML report
    "feedback items": "geri bildirim",
    "themes": "tema",
    "early warnings": "erken uyarı",
    "MRR represented": "Temsil edilen MRR",
    "{items} feedback items · {start} to {end}": "{items} geri bildirim · {start} - {end}",
    "What to work on next": "Sırada ne var",
    ("Points each signal contributes to the priority score (weights: {weights})."): (
        "Her sinyalin öncelik puanına kattığı puan (ağırlıklar: {weights})."
    ),
    "Counting votes vs. weighing evidence": "Oy saymak ve kanıt tartmak",
    (
        "Rank by raw number of mentions (left) vs. Clamor's priority (right). Highlighted lines "
        "moved the most."
    ): (
        "Solda ham bahsedilme sayısına göre sıra, sağda Clamor önceliği. Vurgulanan çizgiler en "
        "çok yer değiştirenler."
    ),
    "How themes moved over time": "Temalar zaman içinde nasıl değişti",
    ("Weekly share of all feedback. Vertical lines mark releases."): (
        "Tüm geri bildirim içindeki haftalık pay. Dikey çizgiler sürümleri gösterir."
    ),
    (
        "Mention rate of the linked theme after vs. before each release, normalized for overall "
        "feedback volume, with 95% intervals. Observational evidence, not an experiment."
    ): (
        "Her sürümden sonra ve önce ilgili temanın bahsedilme oranı, toplam hacme göre "
        "normalleştirilmiş, %95 aralıklarıyla. Gözleme dayalı kanıt, deney değil."
    ),
    ("Generated by Clamor · embedding backend {backend} · as of {date}"): (
        "Clamor ile oluşturuldu · gömme modeli {backend} · {date} itibarıyla"
    ),
    # ---- charts
    "Priority score (points by signal)": "Öncelik puanı (sinyale göre puanlar)",
    "{comp}: {value} pts": "{comp}: {value} puan",
    "Total score": "Toplam puan",
    "By mentions": "Bahsedilmeye göre",
    "By priority": "Önceliğe göre",
    "Rank by mention count": "Bahsedilme sayısına göre sıra",
    "Rank by Clamor priority": "Clamor önceliğine göre sıra",
    "% of feedback": "% geri bildirim",
    " mentions": " bahsedilme",
    "% of all feedback": "Tüm geri bildirimdeki payı (%)",
    "mentions per week": "haftalık bahsedilme",
    "Rate after / before": "Sonra / önce oranı",
    "Mentions before {pre}, after {post}": "Bahsedilme önce {pre}, sonra {post}",
    ("Mention rate after vs before the release (log scale)"): (
        "Sürümden sonra / önce bahsedilme oranı (log ölçek)"
    ),
    # ---- opportunity brief (template)
    "It appeared recently and was absent before.": "Yakın zamanda ortaya çıktı, öncesinde yoktu.",
    "It is **growing fast**": "**Hızla büyüyor**",
    "It is growing": "Büyüyor",
    "It is **declining**": "**Azalıyor**",
    "Its share of feedback is stable": "Geri bildirim içindeki payı durağan",
    ("There is too little recent data to call a trend"): (
        "Eğilim söylemek için yakın dönemde yeterli veri yok"
    ),
    "Trend unknown": "Eğilim bilinmiyor",
    (
        ": the rate over the last {days} days is {ratio}x the baseline (95% CI {lo}-{hi}, FDR q "
        "{q})."
    ): (": son {days} gündeki oran olağan düzeyin {ratio} katı (%95 GA {lo}-{hi}, FDR q {q})."),
    "highest-revenue": "en yüksek gelirli",
    "most frequent": "en sık yazan",
    ("Reproduce with the {people} behind these quotes and add monitoring on the failing path"): (
        "Bu yorumları yazan {people} ile sorunu yeniden üretin ve hatalı akışa izleme ekleyin"
    ),
    ("Ship a targeted fix behind a flag and watch this theme's mention rate afterwards"): (
        "Hedefli bir düzeltmeyi özellik bayrağıyla yayınlayın ve ardından bu temanın bahsedilme "
        "oranını izleyin"
    ),
    ("Proactively tell affected {people} what happened and when it will be fixed"): (
        "Etkilenenlere ne olduğunu ve ne zaman düzeleceğini kendiniz haber verin"
    ),
    ("Interview 5 of the requesting {people} to find the job behind the request"): (
        "İsteğin arkasındaki asıl ihtiyacı bulmak için isteyen {people} arasından 5 kişiyle görüşün"
    ),
    ("Scope the smallest version that unblocks the {key} requesters"): (
        "{key} talep sahiplerinin önünü açacak en küçük sürümü tanımlayın"
    ),
    ("Check whether an integration or workaround covers most of the need today"): (
        "Bir entegrasyonun ya da geçici çözümün ihtiyacın çoğunu bugün karşılayıp karşılamadığını "
        "kontrol edin"
    ),
    ("Run a quick usability test on the flow named in the quotes"): (
        "Yorumlarda geçen akış üzerinde hızlı bir kullanılabilirlik testi yapın"
    ),
    ("Add in-product guidance or better defaults before redesigning the flow"): (
        "Akışı yeniden tasarlamadan önce ürün içi yönlendirme veya daha iyi varsayılanlar ekleyin"
    ),
    ("Instrument the flow to measure where users drop off"): (
        "Kullanıcıların nerede bıraktığını ölçmek için akışa ölçüm ekleyin"
    ),
    ("Segment the complaints by plan and seat count to see who is price sensitive"): (
        "Şikâyetleri plana ve koltuk sayısına göre ayırarak kimin fiyata duyarlı olduğunu görün"
    ),
    ("Test packaging changes (e.g. inactive-seat billing) before list-price changes"): (
        "Liste fiyatını değiştirmeden önce paketleme değişikliklerini (ör. pasif koltuk "
        "faturalaması) test edin"
    ),
    ("Arm customer success with a value narrative for renewal conversations"): (
        "Müşteri başarısı ekibine yenileme görüşmeleri için bir değer anlatısı hazırlayın"
    ),
    "theme {id}": "tema {id}",
    "Problem": "Sorun",
    "Customers report: “{quote}.”": "Müşterilerin ifadesiyle: “{quote}.”",
    "Who is affected": "Kimler etkileniyor",
    (
        "- **{mentions}** mentions from **{n}** {people} in the last {window} days ({total} all "
        "time; {share} of all feedback in that window)"
    ): (
        "- Son {window} günde **{n}** {people} tarafından **{mentions}** kez bahsedildi (tüm "
        "zamanlarda {total}; bu dönemdeki tüm geri bildirimin {share} kadarı)"
    ),
    "- Plans: {plans}": "- Planlar: {plans}",
    "- Channels: {channels}": "- Kanallar: {channels}",
    ("- Accounts raising it represent **{arr} ARR**; revenue-weighted demand is {mrr} MRR"): (
        "- Dile getiren hesapların toplamı **{arr} ARR**; gelir ağırlıklı talep {mrr} MRR"
    ),
    "Evidence": "Kanıtlar",
    "Why now": "Neden şimdi",
    ("\nPriority score **{score}/100**, rank #{rank} (#{votes} if we only counted mentions)."): (
        "\nÖncelik puanı **{score}/100**, sıra #{rank} (yalnızca bahsedilme sayılsaydı #{votes})."
    ),
    ", mention rate x{ratio} afterwards": ", ardından bahsedilme oranı x{ratio}",
    ("\n- Release **{version}** ({date}, {title}): *{verdict}*{extra}"): (
        "\n- **{version}** sürümü ({date}, {title}): *{verdict}*{extra}"
    ),
    "Options to explore": "Değerlendirilecek seçenekler",
    "How we will know it worked": "İşe yaradığını nasıl anlayacağız",
    (
        "- This theme's mention rate drops significantly in Clamor's release radar within 4 weeks "
        "of shipping"
    ): (
        "- Yayından sonraki 4 hafta içinde Clamor'ın sürüm radarında bu temanın bahsedilme oranı "
        "anlamlı biçimde düşer"
    ),
    ("- Sentiment of remaining mentions improves; no new theme spikes after the release"): (
        "- Kalan yorumların duygusu iyileşir; sürümden sonra yeni bir tema sıçraması olmaz"
    ),
    "Open questions": "Açık sorular",
    ("- Which of the quoted {people} can we talk to this week?"): (
        "- Yorumları alıntılanan {people} arasından bu hafta kimlerle konuşabiliriz?"
    ),
    ("- Is there usage data that confirms the size of the problem?"): (
        "- Sorunun büyüklüğünü doğrulayan kullanım verisi var mı?"
    ),
    # ---- dashboard
    ("Customer feedback → prioritized, evidence-backed roadmap"): (
        "Müşteri geri bildirimi → önceliklendirilmiş, kanıta dayalı yol haritası"
    ),
    "Data": "Veri",
    "Demo: Tempo (English B2B SaaS)": "Demo: Tempo (İngilizce B2B SaaS)",
    "Demo: Lezzo (Turkish meal-card app)": "Demo: Lezzo (Türkçe yemek kartı uygulaması)",
    "Upload your own": "Kendi verini yükle",
    ("Feedback: one or more exports (text + date required)"): (
        "Geri bildirim: bir veya daha fazla dışa aktarım (metin + tarih zorunlu)"
    ),
    "Accounts (account_id + mrr, optional)": "Hesaplar (account_id + mrr, isteğe bağlı)",
    "Releases (date + title, optional)": "Sürümler (tarih + başlık, isteğe bağlı)",
    "Language of the feedback": "Geri bildirimin dili",
    "Product name(s) to ignore, comma separated": "Yok sayılacak ürün adları (virgülle ayırın)",
    "Bring your own feedback": "Kendi geri bildiriminizi getirin",
    (
        "Upload a CSV export from your support tool, app store reviews or NPS survey. Only a "
        "**text** column and a **date** column are required; common names such as `body`, "
        "`comment`, `review`, `created`, `timestamp` (and Turkish ones such as `Yorum`, "
        "`Açıklama`, `Tarih`, `Puan`) are recognized automatically, as are Google Play Console "
        "exports. Several files (say, store reviews and support tickets) are combined, each "
        "keeping its own channel. Phone numbers, e-mails, card numbers, IBANs and national ID "
        "numbers are masked before anything is analyzed."
    ): (
        "Destek aracınızdan, uygulama mağazası yorumlarından veya NPS anketinizden bir CSV dışa "
        "aktarımı yükleyin. Yalnızca bir **metin** ve bir **tarih** sütunu gerekir; `Yorum`, "
        "`Açıklama`, `Tarih`, `Puan` gibi Türkçe adlar ve `body`, `comment`, `review`, `created`, "
        "`timestamp` gibi yaygın İngilizce adlar ile Google Play Console dışa aktarımları "
        "otomatik tanınır. Birden fazla dosya (örneğin mağaza yorumları ve destek talepleri) "
        "birleştirilir, her biri kendi kanalını korur. Telefon numaraları, e-postalar, kart "
        "numaraları, IBAN'lar ve TC kimlik numaraları analizden önce maskelenir."
    ),
    (
        "Add an **accounts** file (`account_id`, `mrr`, `plan`) to weigh themes by revenue, and a "
        "**releases** file (`date`, `title`, `description`) to get the release radar."
    ): (
        "Temaları gelire göre tartmak için bir **hesaplar** dosyası (`account_id`, `mrr`, "
        "`plan`), sürüm radarı için de bir **sürümler** dosyası (`date`, `title`, `description`) "
        "ekleyin."
    ),
    (
        "Files are processed in memory by the server running this app and are not stored. For "
        "confidential data, run the app on your own machine: then nothing leaves it except, if "
        "you enable it, theme summaries sent to Claude."
    ): (
        "Dosyalar bu uygulamayı çalıştıran sunucuda bellekte işlenir ve saklanmaz. Gizli veriler "
        "için uygulamayı kendi bilgisayarınızda çalıştırın: o zaman, açarsanız Claude'a "
        "gönderilen tema özetleri dışında hiçbir şey bilgisayarınızdan çıkmaz."
    ),
    "Could not read the input: {error}": "Girdi okunamadı: {error}",
    "Analyze as of": "Analiz tarihi",
    ("Time travel: see what Clamor would have told you on that day."): (
        "Zamanda yolculuk: Clamor'ın o gün size ne söyleyeceğini görün."
    ),
    "What matters most?": "En önemli olan ne?",
    "Weight preset": "Ağırlık ön ayarı",
    "Fine-tune weights": "Ağırlıkları ince ayarla",
    "Reach (accounts)": "Erişim (hesap sayısı)",
    "Revenue-weighted demand": "Gelir ağırlıklı talep",
    "Severity (negative sentiment)": "Şiddet (olumsuz duygu)",
    "Momentum (significant growth)": "İvme (anlamlı artış)",
    "Model": "Model",
    "Embedding backend": "Gömme modeli",
    (
        "Default for this language: {default}. {semantic} = semantic sentence embeddings; hybrid "
        "adds TF-IDF vocabulary; tfidf needs no model download."
    ): (
        "Bu dil için varsayılan: {default}. {semantic} = anlamsal cümle gömmeleri; hybrid bunlara "
        "TF-IDF kelime bilgisini ekler; tfidf model indirmeyi gerektirmez."
    ),
    "Review themes with Claude": "Temaları Claude ile gözden geçir",
    ("Names, classifies and de-duplicates themes. Needs ANTHROPIC_API_KEY."): (
        "Temalara ad verir, sınıflandırır ve tekrarları birleştirir. ANTHROPIC_API_KEY gerekir."
    ),
    ("Set `ANTHROPIC_API_KEY` to enable the Claude analyst layer."): (
        "Claude analist katmanını açmak için `ANTHROPIC_API_KEY` tanımlayın."
    ),
    "Discovering themes (embedding and clustering)...": "Temalar bulunuyor (gömme ve kümeleme)...",
    "Claude is reviewing the themes...": "Claude temaları gözden geçiriyor...",
    ("Backtesting early warnings day by day..."): (
        "Erken uyarılar gün gün geriye dönük test ediliyor..."
    ),
    "Claude unavailable: {error}": "Claude kullanılamıyor: {error}",
    (
        "The `{backend}` model could not be loaded, so Clamor fell back to `{fallback}`. Results "
        "will be less accurate."
    ): (
        "`{backend}` modeli yüklenemediği için Clamor `{fallback}` modeline geçti. Sonuçlar daha "
        "az isabetli olacak."
    ),
    "What should we build next?": "Sırada ne yapmalıyız?",
    "Feedback items": "Geri bildirim",
    "Themes": "Temalar",
    "Revenue data": "Gelir verisi",
    "not provided": "yok",
    "Roadmap": "Yol haritası",
    "Theme explorer": "Tema gezgini",
    "Early warning": "Erken uyarı",
    "Model quality": "Model kalitesi",
    "Priority ranking": "Öncelik sıralaması",
    (
        "Points each signal contributes. Weights: reach {reach}, revenue {revenue}, severity "
        "{severity}, momentum {momentum}. Reach, revenue and severity use the last {days} days."
    ): (
        "Her sinyalin kattığı puan. Ağırlıklar: erişim {reach}, gelir {revenue}, şiddet "
        "{severity}, ivme {momentum}. Erişim, gelir ve şiddet son {days} günü kullanır."
    ),
    (
        "Left: rank by raw number of mentions. Right: Clamor's priority. Highlighted themes moved "
        "the most."
    ): (
        "Solda ham bahsedilme sayısına göre sıra, sağda Clamor önceliği. Vurgulanan temalar en "
        "çok yer değiştirenler."
    ),
    "{kind} · keywords: {keywords}": "{kind} · anahtar kelimeler: {keywords}",
    "Mentions ({days}d)": "Bahsedilme ({days} gün)",
    "Sentiment": "Duygu",
    "Trend: mention rate over the last {days} days vs. the baseline.": (
        "Eğilim: son {days} gündeki bahsedilme oranının olağan düzeye oranı."
    ),
    "What customers say": "Müşteriler ne diyor",
    "share": "pay",
    "Plan mix": "Plan dağılımı",
    "Opportunity brief": "Fırsat özeti",
    "Write it with Claude": "Claude ile yaz",
    "Download brief (Markdown)": "Özeti indir (Markdown)",
    "Evidence pack sent to the brief writer": "Özet yazarına gönderilen kanıt paketi",
    (
        "Each theme's mention rate in the last {recent} days vs. the {baseline} days before, "
        "normalized for overall feedback volume (median-of-ratios), tested with an exact Poisson "
        "rate test and corrected for multiple comparisons (Benjamini-Hochberg)."
    ): (
        "Her temanın son {recent} gündeki bahsedilme oranı, önceki {baseline} günle "
        "karşılaştırılır; toplam geri bildirim hacmine göre normalleştirilir (oranların medyanı), "
        "kesin Poisson oran testiyle sınanır ve çoklu karşılaştırma için düzeltilir "
        "(Benjamini-Hochberg)."
    ),
    (
        "Tip: move **Analyze as of** in the sidebar to mid-March to watch the sync regression get "
        "flagged."
    ): (
        "İpucu: senkronizasyon sorununun nasıl yakalandığını görmek için kenar çubuğundaki "
        "**Analiz tarihi**ni Mart ortasına getirin."
    ),
    (
        "Tip: move **Analyze as of** in the sidebar to late March to watch the login problems "
        "after v5.1 get flagged."
    ): (
        "İpucu: v5.1 sonrasındaki giriş sorunlarının nasıl yakalandığını görmek için kenar "
        "çubuğundaki **Analiz tarihi**ni Mart sonuna getirin."
    ),
    "Recent": "Son dönem",
    "Baseline": "Olağan düzey",
    "95% CI low": "%95 GA alt",
    "95% CI high": "%95 GA üst",
    "q-value (FDR)": "q değeri (FDR)",
    "Compare themes over time (up to 4)": "Temaları zaman içinde karşılaştır (en fazla 4)",
    (
        "Upload a releases file (date, title, description) to see whether shipped work changed "
        "what customers talk about."
    ): (
        "Yayınlanan işlerin müşterilerin konuştuklarını değiştirip değiştirmediğini görmek için "
        "bir sürümler dosyası (tarih, başlık, açıklama) yükleyin."
    ),
    (
        "For each release, the linked theme's mention rate after vs. before (windows cut at "
        "neighbouring releases, normalized for overall volume). Observational evidence, not a "
        "controlled experiment."
    ): (
        "Her sürüm için ilgili temanın sonraki ve önceki bahsedilme oranı (pencereler komşu "
        "sürümlerde kesilir, toplam hacme göre normalleştirilir). Gözleme dayalı kanıt, kontrollü "
        "deney değil."
    ),
    ("**Suspected side effects**: themes that spiked after a release they were not linked to."): (
        "**Olası yan etkiler**: bağlı olmadıkları bir sürümden sonra sıçrayan temalar."
    ),
    "Before": "Önce",
    "After": "Sonra",
    (
        "Model quality is measured on the synthetic dataset, where the true theme of every item "
        "is known. Switch to the demo data to see it."
    ): (
        "Model kalitesi, her öğenin gerçek temasının bilindiği sentetik veride ölçülür. Görmek "
        "için demo verisine geçin."
    ),
    ("Move **Analyze as of** to the last day to run the evaluation."): (
        "Değerlendirmeyi çalıştırmak için **Analiz tarihi**ni son güne getirin."
    ),
    "Theme purity (homogeneity)": "Tema saflığı (homojenlik)",
    "Releases linked correctly": "Doğru bağlanan sürümler",
    ("**Early-warning backtest**: history replayed day by day."): (
        "**Erken uyarı geriye dönük testi**: geçmiş gün gün yeniden oynatıldı."
    ),
    "Naive: week ≥ 2x average": "Basit kural: hafta ≥ ortalamanın 2 katı",
    "How well each true theme was recovered": "Her gerçek tema ne kadar iyi bulundu",
    "Clamor · open source · synthetic demo data": "Clamor · açık kaynak · sentetik demo verisi",
}


def t(text: str, lang: str = "en", **values) -> str:
    """Translate an English string (a ``str.format`` template when values are given)."""
    out = TR.get(text, text) if lang == "tr" else text
    return out.format(**values) if values else out


def number(x: float, lang: str = "en", digits: int = 0) -> str:
    """1,234.5 in English, 1.234,5 in Turkish."""
    s = f"{x:,.{digits}f}"
    return s.translate(str.maketrans(",.", ".,")) if lang == "tr" else s


def pct(x: float, lang: str = "en", digits: int = 0, signed: bool = False) -> str:
    """94% in English, %94 in Turkish (the sign goes first: +%310)."""
    body = number(abs(x) * 100 if signed else x * 100, lang, digits)
    sign = ("+" if x >= 0 else "-") if signed else ""
    return f"{sign}%{body}" if lang == "tr" else f"{sign}{body}%"


def q_value(q: float, lang: str = "en") -> str:
    """'< 0.001' instead of '0.000' (see :func:`clamor.stats.format_q`)."""
    return f"< {number(0.001, lang, 3)}" if q < 0.001 else number(q, lang, 3)


def date(value, lang: str = "en") -> str:
    """Mar 23, 2026 in English, 23 Mart 2026 in Turkish."""
    d = pd.Timestamp(value)
    return f"{d.day} {TR_MONTHS[d.month - 1]} {d.year}" if lang == "tr" else f"{d:%b %d, %Y}"


def money(x: float, lang: str = "en", compact: bool = False) -> str:
    """$34,500 or, compact, $34.5k / $1.6M (Turkish: $34.500, $34,5 bin, $1,6 mn)."""
    if compact and x >= 1_000_000:
        return f"${number(x / 1_000_000, lang, 1)}" + (" mn" if lang == "tr" else "M")
    if compact and x >= 1000:
        return f"${number(x / 1000, lang, 1)}" + (" bin" if lang == "tr" else "k")
    return f"${number(x, lang)}"


def people(word: str, lang: str = "en", form: str = "count") -> str:
    """'accounts' or 'users' (see ``Analysis.people``) in the output language.

    ``form``: "count" follows a number ("74 users"; Turkish keeps it singular, "74
    kullanıcı"), "plural" stands alone ("kullanıcılar"), "title" is a column heading.
    """
    if form == "title":
        return t(word.capitalize(), lang)
    if form == "plural" and lang == "tr":
        return t(f"{word} (plural)", lang)
    return t(word, lang)


def channel(name: str, lang: str = "en") -> str:
    """support_ticket -> 'support ticket' / 'destek talebi'."""
    return t(str(name).replace("_", " "), lang)
