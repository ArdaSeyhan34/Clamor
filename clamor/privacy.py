"""Masking personal data before anything is analyzed, stored in a report or sent to an API.

Support tickets routinely contain phone numbers, e-mail addresses and, in payments, card
numbers and IBANs. Clamor masks them on load (``Config.redact_pii``, on by default), so
they never reach the embeddings, the quotes in reports or the optional Claude layer.

Numbers are only masked when they look like the real thing: card numbers must pass the
Luhn check and Turkish national ID numbers (TCKN) their checksum, so order totals, dates
and amounts such as "150 TL" are left alone.
"""

from __future__ import annotations

import re

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){3,7}(?:[ ]?[A-Z0-9]{1,3})?\b")
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
_TCKN = re.compile(r"(?<!\d)[1-9]\d{10}(?!\d)")
_PHONE_TR = re.compile(r"(?<!\d)(?:\+?90[ -]?)?\(?0?5\d{2}\)?[ -]?\d{3}[ -]?\d{2}[ -]?\d{2}(?!\d)")
_PHONE_INTL = re.compile(r"(?<!\w)\+\d{1,3}[ -]?(?:\d[ -]?){6,12}\d(?!\d)")


def luhn_valid(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def tckn_valid(number: str) -> bool:
    """Checksum of the Turkish national ID number (11 digits, first digit not 0)."""
    if len(number) != 11 or not number.isdigit() or number[0] == "0":
        return False
    d = [int(c) for c in number]
    tenth = ((d[0] + d[2] + d[4] + d[6] + d[8]) * 7 - (d[1] + d[3] + d[5] + d[7])) % 10
    return tenth == d[9] and sum(d[:10]) % 10 == d[10]


def redact(text: str) -> str:
    """Replace personal data with placeholders such as ``[card]`` or ``[phone]``."""
    if not isinstance(text, str) or not text:
        return text
    text = _EMAIL.sub("[email]", text)
    text = _IBAN.sub("[iban]", text)

    def card(m: re.Match) -> str:
        digits = re.sub(r"\D", "", m.group())
        return "[card]" if 13 <= len(digits) <= 19 and luhn_valid(digits) else m.group()

    text = _CARD.sub(card, text)
    text = _TCKN.sub(lambda m: "[national-id]" if tckn_valid(m.group()) else m.group(), text)
    text = _PHONE_TR.sub("[phone]", text)
    text = _PHONE_INTL.sub("[phone]", text)
    return text
