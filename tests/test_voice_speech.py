"""VOICE_PLAN rule 4: the speech normaliser, both languages, on the shapes the agent really writes (eval transcripts of 19 Oct)."""

from falcon.services.voice.speech import amount_words, normalise, number_words, spell


def test_number_words_en_and_ar():
    assert number_words(1291, "en") == "one thousand two hundred ninety one"
    assert number_words(842, "en") == "eight hundred forty two"
    assert number_words(1291, "ar") == "ألف ومئتان وواحد وتسعون"
    assert number_words(100, "ar") == "مائة"


def test_codes_are_spelled_character_by_character():
    assert spell("K3P7QZ", "en") == "K three P seven Q Z"
    assert spell("FA929", "en") == "F A nine two nine"
    assert spell("CR-K3P7QZ", "en") == "C R, K three P seven Q Z"
    assert spell("K3P7QZ", "ar") == "كيه ثلاثة بي سبعة كيو زد"
    assert spell("FA929", "ar") == "إف إيه تسعة اثنان تسعة"


def test_amounts_with_and_without_minor_units():
    assert amount_words("1,291.00", "en") == "one thousand two hundred ninety one dirhams"
    assert amount_words("1.00", "en") == "one dirham"
    assert amount_words("12.50", "en") == "twelve dirhams and fifty fils"
    assert amount_words("1291.00", "ar") == "ألف ومئتان وواحد وتسعون درهم"


def test_english_reply():
    text = "Booking K3P7QZ — Dubai (DXB) to Cairo (CAI), flight FA929, total paid AED 1,391.00."
    assert normalise(text, "en") == ("Booking K three P seven Q Z, Dubai (D X B) to Cairo (C A I), flight F A nine two nine, "
                                     "total paid one thousand three hundred ninety one dirhams.")


def test_english_clause_route_and_markdown():
    text = "Your booking A9C3PN (FA851, DXB–KWI). See **clause 7.4.4** and 9.1.2."
    assert normalise(text, "en") == ("Your booking A nine C three P N (F A eight five one, D X B to K W I). "
                                     "See clause seven point four point four and nine point one point two.")


def test_arabic_cancellation_list_becomes_sentences():
    text = "تم إلغاء الحجز K3P7QZ بنجاح.\n\n- رسوم الإلغاء: 100.00 AED\n- المبلغ المسترد: 1291.00 AED كرصيد سفر"
    assert normalise(text, "ar") == ("تم إلغاء الحجز كيه ثلاثة بي سبعة كيو زد بنجاح. رسوم الإلغاء: مائة درهم. "
                                     "المبلغ المسترد: ألف ومئتان وواحد وتسعون درهم كرصيد سفر.")


def test_arabic_clause_date_and_flight():
    text = "تنص المادة 7.3.2 على ذلك، والمهلة انتهت في 2026-10-20 02:30 (FA811 من DXB إلى RUH)."
    assert normalise(text, "ar") == ("تنص المادة سبعة نقطة ثلاثة نقطة اثنان على ذلك، والمهلة انتهت في 20 أكتوبر 2026 02:30 "
                                     "(إف إيه ثمانية واحد واحد من دي إكس بي إلى آر يو إتش).")


def test_arabic_amount_written_with_the_word_dirham():
    assert normalise("يُقيَّد لك مبلغ 942.00 درهم كرصيد سفر", "ar") == "يُقيَّد لك مبلغ تسعمائة واثنان وأربعون درهم كرصيد سفر."


def test_emoji_headers_and_links_are_dropped():
    assert normalise("## Your trip\n✈ [details](https://x.example/a) ready", "en") == "Your trip. details ready."
