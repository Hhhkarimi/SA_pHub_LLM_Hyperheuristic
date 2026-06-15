# هایپرهیوریستیک مبتنی بر مدل زبانی برای مسئله مکان‌یابی p-هاب تک‌تخصیصی

این مخزن شامل کد پایتون نهایی و داده‌های CAB برای روش هایپرهیوریستیک مبتنی بر مدل زبانی در مسئله مکان‌یابی p-هاب تک‌تخصیصی است.

## محتویات

- `sa_p_hub_llm_hyperheuristic.py`: کد نهایی حل مسئله.
- `CAB_instances/`: داده‌های CAB در اندازه‌های ۵، ۱۰، ۱۵، ۲۰ و ۲۵ گره.
- `CAB_reference_optima.csv`: مقادیر بهینه مرجع برای محاسبه شکاف.
- `Modelfile.phi3-cab.v2`: نمونه فایل مدل برای اجرای محلی Phi-3 با Ollama.
- `requirements.txt`: وابستگی‌های پایتون.

## خلاصه روش

مدل زبانی به‌عنوان حل‌کننده مستقیم استفاده نمی‌شود. در هر تکرار، مدل یک دستور فشرده مانند زیر تولید می‌کند:

```text
W=0.35,0.18,0.30,0.10,0.07;R=3;A=none
```

بردار `W` وزن معیارهای جریان، مرکزیت، پراکندگی، تعامل جریانی و جریمه فاصله را مشخص می‌کند. مقدار `R` اندازه فهرست کاندیدای محدود است و `A` فهرست اختیاری اجتناب موقت از هاب‌های پرتکرار را نشان می‌دهد. ساخت جواب، اصلاح تخصیص، کنترل امکان‌پذیری و محاسبه تابع هدف توسط کد پایتون انجام می‌شود.

## نصب

```bash
python -m venv .venv
source .venv/bin/activate  # در ویندوز: .venv\Scripts\activate
pip install -r requirements.txt
```

برای اجرای محلی مدل زبانی، Ollama را نصب کنید و مدل محلی خود را بسازید:

```bash
ollama create phi3-cab-v2 -f Modelfile.phi3-cab.v2
```

فایل مدل GGUF در این مخزن قرار داده نشده است.

## نمونه اجرا

```bash
python sa_p_hub_llm_hyperheuristic.py --n 10 --p 2 --alpha 0.8 --model phi3-cab-v2 --llm-provider ollama
```

نمایش پرامپت ارسال‌شده به مدل:

```bash
python sa_p_hub_llm_hyperheuristic.py --n 10 --p 2 --alpha 0.8 --model phi3-cab-v2 --llm-provider ollama --print-prompt
```

اجرای بدون مدل زبانی:

```bash
python sa_p_hub_llm_hyperheuristic.py --n 10 --p 2 --alpha 0.8 --no-llm
```
