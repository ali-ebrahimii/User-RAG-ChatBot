import re
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

FALLBACK_CONTACT_HTML = (
    '<p dir="rtl" lang="fa">'
    'این سوال خارج از حیطه تحصصی ما می باشد.'
    'در صورت اضطرار می توانید برای پاسخ این سوال با شماره تلفن <strong>۰۲۱۹۱۰۹۸۰۹۰</strong> تماس بگیرید.'
    '</p>'
)

MEDICAL_REDIRECT_HTML = (
    '<article dir="rtl" lang="fa">'
    '<header><h2>راهنمای دریافت مشاوره پزشکی</h2></header>'
    '<p>این سامانه برای <strong>ویزیت آنلاین</strong> و مدیریت خدمات «سامان سلامت» است و ارائه <strong>توصیه درمانی مستقیم</strong> در چت انجام نمی‌شود.</p>'
    '<p>لطفاً از طریق «سامان سلامت» یک <strong>ویزیت آنلاین</strong> ثبت کنید تا پزشک راهنمایی‌تان کند. '
    'در صورت علائم اورژانسی مثل <strong>درد قفسه سینه</strong>، <strong>تنگی نفس</strong> یا <strong>بیهوشی</strong> با <strong>۱۱۵</strong> تماس بگیرید.</p>'
    '</article>'
)

@dataclass
class IntentResult:
    intent: str
    confidence: float
    anchor_query: Optional[str] = None
    notes: Optional[str] = None

_FA_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED]")

def normalize_fa(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    t = t.replace("ي", "ی").replace("ك", "ک")
    t = t.replace("\u200c", " ")
    t = _FA_DIACRITICS.sub("", t)
    t = re.sub(r"\s+", " ", t)
    return t


def _contains_any(t: str, words: List[str]) -> bool:
    return any(w in t for w in words)

_MEDICAL_WORDS = [
    "علائم", "درد", "تب", "سرفه", "تنگی نفس", "خونریزی", "سکته", "حمله قلبی",
    "دارو", "قرص", "آنتی بیوتیک", "دوز", "عوارض", "تشخیص", "درمان", "بیماری",
    "بارداری", "فشار خون", "قند خون", "دیابت", "سرطان", "بیهوشی",
]

_OUT_SCOPE_WORDS = [
    "شعر", "جوک", "ترجمه", "کد بزن", "پایتون", "بیت کوین", "کریپتو",
    "فوتبال", "هواشناسی", "آشپزی", "فیلم", "سریال"
]

_ONBOARDING_PATTERNS = [
    r"از\s*کجا\s*(باید\s*)?شروع",
    r"چطور\s*شروع",
    r"چجوری\s*شروع",
    r"اولین\s*(قدم|مرحله)",
    r"تازه\s*(وارد|ثبت\s*نام|عضو)\s*شدم",
    r"(راهنمایی|کمک)\s*(میخوام|می‌خوام)?",
    r"نمی\s*دونم\s*چیکار\s*کنم",
    r"چیکار\s*کنم",
]

# Auth / login
_AUTH_WORDS = ["ورود", "لاگین", "ثبت نام", "ثبت‌نام", "کد پیامکی", "otp", "رمز یکبارمصرف", "رمز عبور", "فراموش"]

# Appointment / booking
_BOOKING_WORDS = ["نوبت", "ویزیت", "پزشک", "زمان", "رزرو", "دریافت نوبت", "لیست پزشکان"]

# Cancel / edit appointment
_CANCEL_WORDS = ["لغو", "کنسل", "تغییر", "جابجا", "ویرایش", "تعویض", "نوبتم"]

# Payment
_PAY_WORDS = ["پرداخت", "هزینه", "قیمت", "مبلغ", "درگاه", "تراکنش", "رسید", "فرانشیز"]

# E-prescription / social security
_EPRES_WORDS = ["نسخه الکترونیک", "تامین اجتماعی", "کد رهگیری", "ep.tamin", "داروخانه", "نسخه نویسی"]

# Support / contact
_SUPPORT_WORDS = ["پشتیبانی", "تماس", "ساعت پاسخگویی", "گفتگو با پشتیبانی", "شماره", "چت"]

# Privacy / terms
_PRIVACY_WORDS = ["حریم خصوصی", "پرایوسی", "قوانین", "شرایط", "terms", "agreement", "محرمانگی", "اطلاعات شخصی"]

# Insurance (if relevant in your product)
_INS_WORDS = ["بیمه", "بیمه تکمیلی", "خسارت", "معرفی نامه", "پرتال درمانت", "دِرمانت", "پرداخت خسارت"]


def classify_intent_fa(q: str) -> str:
    """
    Backward compatible API: returns only intent string.
    """
    return classify_intent_fa_v2(q).intent


def classify_intent_fa_v2(q: str) -> IntentResult:
    t = normalize_fa(q)
    if not t:
        return IntentResult(intent="empty", confidence=1.0)

    # 1) Medical (highest priority)
    if _contains_any(t, _MEDICAL_WORDS):
        return IntentResult(intent="medical_advice", confidence=0.95)

    # 2) Hard out-of-scope
    if _contains_any(t, _OUT_SCOPE_WORDS):
        return IntentResult(intent="out_of_scope", confidence=0.90)

    # 3) Onboarding/start/help (ambiguous)
    if any(re.search(pat, t) for pat in _ONBOARDING_PATTERNS):
        # Anchor to user-flow / booking entry (most common starting point)
        return IntentResult(
            intent="onboarding_start",
            confidence=0.85,
            anchor_query="نحوه دریافت نوبت سایت سامان سلامت saman.health ورود شماره تلفن رمز یکبارمصرف انتخاب پزشک انتخاب زمان نوبت های من"
        )

    # 4) Auth/login
    if _contains_any(t, _AUTH_WORDS):
        return IntentResult(
            intent="auth_login",
            confidence=0.80,
            anchor_query="ورود به سامانه کد پیامکی رمز یکبارمصرف فراموشی رمز"
        )

    # 5) Booking
    if _contains_any(t, _BOOKING_WORDS):
        # if also cancel/edit terms exist, prioritize cancel
        if _contains_any(t, _CANCEL_WORDS):
            return IntentResult(
                intent="appointment_cancel_edit",
                confidence=0.85,
                anchor_query="لغو نوبت تغییر زمان ویرایش نوبت نوبت های من"
            )
        return IntentResult(
            intent="appointment_booking",
            confidence=0.85,
            anchor_query="مسیر دریافت نوبت ویزیت آنلاین لیست پزشکان انتخاب زمان"
        )

    # 6) Payment
    if _contains_any(t, _PAY_WORDS):
        return IntentResult(
            intent="payment_billing",
            confidence=0.80,
            anchor_query="پرداخت هزینه فرانشیز رسید تراکنش"
        )

    # 7) E-prescription
    if _contains_any(t, _EPRES_WORDS):
        return IntentResult(
            intent="e_prescription",
            confidence=0.80,
            anchor_query="نسخه الکترونیک تامین اجتماعی کد رهگیری مشاهده نسخه"
        )

    # 8) Privacy/Terms
    if _contains_any(t, _PRIVACY_WORDS):
        return IntentResult(
            intent="privacy_terms",
            confidence=0.80,
            anchor_query="حریم خصوصی شرایط و قوانین مسئولیت کاربران"
        )

    # 9) Support
    if _contains_any(t, _SUPPORT_WORDS):
        return IntentResult(
            intent="support_contact",
            confidence=0.80,
            anchor_query="پشتیبانی شماره تماس ساعات پاسخگویی گفتگو با پشتیبانی"
        )

    # 10) Insurance-ish
    if _contains_any(t, _INS_WORDS):
        return IntentResult(
            intent="insurance_info",
            confidence=0.75,
            anchor_query="بیمه تکمیلی معرفی نامه خسارت درمانت پرتال"
        )

    # Default: allow RAG to try, but keep as generic_faq (your current behavior)
    return IntentResult(intent="green_policy_faq", confidence=0.60)

