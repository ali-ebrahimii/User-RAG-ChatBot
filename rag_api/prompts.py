import re
from typing import List, Dict
from .config import MAX_CTX_CHARS


def build_messages(question: str, hits: List[Dict]) -> List[Dict]:
    context = "\n\n---\n\n".join([h["text"] for h in hits])
    if len(context) > MAX_CTX_CHARS:
        context = context[:MAX_CTX_CHARS]

    fallback_html = (
        '<article dir="rtl" lang="fa">'
        '<p>برای پاسخ این سوال با شماره تلفن <strong>۰۲۱۹۱۰۹۸۰۹۰</strong> تماس بگیرید.</p>'
        "</article>"
    )

    system = (
        "تو دستیار رسمی پلتفرم «سامان سلامت» هستی.\n"
        "فقط و فقط بر اساس «منابع» داده‌شده پاسخ بده و از دانش بیرونی استفاده نکن.\n"
        "\n"
        "خروجی باید HTML معتبر باشد و فقط یک پاسخ بدهی.\n"
        "خروجی باید دقیقاً با این پیشوند شروع شود:\n"
        "ANSWER: <HTML>\n"
        "\n"
        "قواعد بسیار مهم:\n"
        "1) فقط یک پاسخ تولید کن و هیچ توضیحی خارج از HTML ننویس.\n"
        "2) HTML فقط داخل ANSWER باشد (بدون Markdown).\n"
        "3) HTML را برای فارسی تنظیم کن: <article dir=\"rtl\" lang=\"fa\"> ... </article>\n"
        "4) از تگ‌های معنایی استفاده کن: <article>, <header>, <h2>, <p>, <ul>, <li>.\n"
        "5) نکات مهم را با <strong> مشخص کن.\n"
        "6) پاسخ کوتاه باشد: حداکثر 2 پاراگراف یا 1 لیست با حداکثر 4 آیتم.\n"
        "\n"
        "لطفا غلط املایی نداشته باش و روان بنویس\n"
        "\n"
        "محدودیت دامنه (خیلی مهم):\n"
        "- فقط درباره خدمات، قوانین، پرداخت، حساب کاربری، حریم خصوصی، EHR و ویزیت آنلاینِ «سامان سلامت» پاسخ بده.\n"
        "- اگر سوال خارج از موضوع سامان سلامت بود (مثل هواشناسی، اخبار، ورزش، سوال عمومی)، یا پاسخ دقیق در منابع نبود،\n"
        f"  دقیقاً فقط این را بده:\nANSWER: {fallback_html}\n"
    )

    user = f"سوال:\n{question}\n\nمنابع:\n{context}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

def normalize_answer_html(html: str) -> str:
    html = (html or "").strip()

    # remove markdown fences
    html = re.sub(r"^```(?:html)?\s*", "", html, flags=re.IGNORECASE)
    html = re.sub(r"\s*```$", "", html)

    if not html:
        return '<article dir="rtl" lang="fa"><p>متأسفانه پاسخی یافت نشد.</p></article>'

    # wrap if not already wrapped
    if "<article" not in html.lower():
        html = f'<article dir="rtl" lang="fa">{html}</article>'

    return html

def extract_answer_line(generated_text: str) -> str:
    txt = (generated_text or "").strip()
    m = re.search(r"ANSWER\s*:\s*(.*)", txt, flags=re.DOTALL)
    if m:
        html = m.group(1).strip()
    else:
        html = txt.strip()

    html = normalize_answer_html(html)
    return f"ANSWER: {html}"
