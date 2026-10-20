"""Decision #55: the STT comparison counts errors after one normalisation for every service."""

from falcon.services.voice.wer import error_rates, normalise


def test_spelling_style_is_not_an_error():
    assert normalise("Booking K 3 P 7 Q Z, please!", "en") == ["booking", "k3p7qz", "please"]
    assert normalise("حقيبة وزنها 23 كيلو", "ar") == normalise("حقيبه وزنها ثلاثة وعشرون كيلو", "ar")
    assert normalise("أُفضّل الاسترداد", "ar") == normalise("افضل الاسترداد", "ar")


def test_error_rates():
    assert error_rates("Yes, please go ahead.", "yes please go ahead", "en") == (0.0, 0.0)
    wer, cer = error_rates("I want to cancel my booking", "I want to council my booking", "en")
    assert round(wer, 3) == round(1 / 6, 3) and 0 < cer < wer
    assert error_rates("نعم، أؤكد الإلغاء.", "", "ar")[0] == 1.0


def test_english_spelling_style_of_codes_and_contractions():
    # 21 Oct reading call: "I would like ... K3P7Q zed", "FA 929" are the script's "I'd like ... K3P7QZ", "FA929"
    assert normalise("Hi. I would like to cancel my booking, K3P7Q zed.", "en") == normalise("Hi, I'd like to cancel my booking K3P7QZ.", "en")
    assert normalise("My flight is FA 929 from Dubai", "en") == normalise("My flight is FA929 from Dubai", "en")
    assert normalise("about 23 kilos", "en") == ["about", "twenty", "three", "kilos"]
