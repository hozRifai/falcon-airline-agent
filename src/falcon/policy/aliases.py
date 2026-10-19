"""Customer words -> the contract's words, appended to the query before it is embedded (decision #39, measured in docs/EVALUATION.md §1).

The two standing misses of the retrieval gold set were vocabulary gaps no rewrite closed: "I missed my outbound flight, is my return
still valid?" is, in the Conditions, "cancellation of onward reservations" and "fail to check-in" (5.7.1); "cancellation due to a medical
emergency" is "Force Majeure" and "voluntary refund" (3.3.1, 10.3.1). A hand-written list is honest about what it knows: every entry is
a phrase a customer used and the phrase the document uses for it, in both languages where the customer may write in Arabic.
"""

from __future__ import annotations

import re

ALIASES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"missed my (?:outbound |first |onward )?flight|no.?show|didn'?t (?:check in|show up|board)|miss(?:ed)? (?:the )?check.?in|فاتتني رحل|لم أسجل|لم أصل في الوقت", re.I),
     "fail to check-in Check-in Deadline cancellation of onward reservations unused Flight Coupons sequence"),
    (re.compile(r"medical emergenc|illness|\bill\b|sick|hospital|surgery|doctor says|unable to travel|cannot travel|prevented from travel|حالة طبية|مرض|مستشفى|عملية جراحية|لا أستطيع السفر", re.I),
     "Force Majeure prevented from travelling non-refundable fare credit adequate evidence voluntary refund"),
    (re.compile(r"passed away|died|death|funeral|bereave|توفي|وفاة|جنازة", re.I),
     "member of your Immediate Family dies death certificate refund before beginning your carriage"),
    (re.compile(r"overbook|bumped|wouldn'?t let me board|denied boarding|refused boarding|flight was full|رفضوا صعودي|الرحلة ممتلئة", re.I),
     "denied boarding confirmed reservation later flight different class involuntary refund compensation"),
    (re.compile(r"refund (?:to|on) my card|back (?:to|on) my card|which currency|what currency|إلى بطاقتي|بأي عملة", re.I),
     "refund in the same form and currency used to pay for the Ticket to the person who paid"),
    (re.compile(r"compensat|hotel|meal|accommodation|delay(?:ed)? (?:for|by)|تعويض|فندق|وجبة", re.I),
     "remedies for cancellation rerouting and delays liability for damage caused by delay"),
]


def expand(query: str) -> tuple[str, list[str]]:
    """(query with the matching contract phrases appended, the phrases used). No match: the query as typed."""
    added = [phrase for pattern, phrase in ALIASES if pattern.search(query)]
    added = list(dict.fromkeys(added))
    return (query + " | " + " ; ".join(added)) if added else query, added
