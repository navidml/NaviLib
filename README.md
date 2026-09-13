# NaviLib

کتابخانهٔ کمکی تحلیل داده و یادگیری ماشین برای داده‌های جدولی؛ با خروجی‌های قابل بررسی، پیش‌پردازش قابل بازاجرا، نمودارهای هماهنگ و گزارش HTML مستقل.

```bash
python -m pip install ".[dev,notebook]"
```

این دستور را داخل پوشهٔ پروژه اجرا کنید. Python 3.10 یا جدیدتر لازم است. نام توزیع `NaviLib` و نام import نیز دقیقاً `NaviLib` است؛ مثال‌های قدیمی با نام `navdata` یا `datakit` مربوط به این پکیج نیستند.

```python
import pandas as pd
import NaviLib as nv

df = pd.DataFrame({
    "age": [24, 31, None, 42, 29],
    "city": ["Tehran", "Shiraz", "Tehran", "Tabriz", "Shiraz"],
})

nv.set_theme("light")
issues = nv.audit_data(df)
report = nv.create_report(df, title="Customer data review")
report.to_html("outputs/report.html")

nv.help_map("missing")       # جست‌وجوی تابع، امضا، توضیح و نام جایگزین
help(nv.impute_missing)      # راهنمای پارامترها و خروجی
```

گزارش HTML بدون اینترنت باز می‌شود و شامل جدول‌های کیفیت، پیشنهاد اقدام، آمار توصیفی و نمودار است. دادهٔ خام را تغییر نمی‌دهد. جدول‌های کامل در `report.tables` و شکل‌ها در `report.figures` در دسترس‌اند؛ نمایش HTML تعداد ردیف‌ها را محدود می‌کند. گزارش، داده و نمودارهای تعبیه‌شده را در خود دارد؛ فایل نمونهٔ این پروژه فقط از دادهٔ ساختگی استفاده می‌کند.

## مسیرهای اصلی

| ماژول | کاربرد |
|---|---|
| `cleaning` | بررسی داده، نام ستون‌ها، مقادیر گمشده، پرت‌ها، تکرارها، تقسیم و توازن کلاس‌ها |
| `eda` | آمار توصیفی، ارتباط ویژگی و هدف، همبستگی، تغییر توزیع و نمودارها |
| `feature_engineering` | تبدیل عددی، مقیاس‌بندی، کدگذاری، بازه‌بندی، ویژگی تاریخ و متن و تعامل‌ها |
| `statistical_tests` | آزمون‌های پارامتریک و ناپارامتریک، اثر، همبستگی و اصلاح آزمون‌های متعدد |
| `modeling` | پایپ‌لاین، اعتبارسنجی، جست‌وجوی پارامتر، آموزش، توضیح و ذخیرهٔ مدل |
| `evaluation` | ارزیابی طبقه‌بندی، رگرسیون، خوشه‌بندی و رتبه‌بندی |
| `quality` | بررسی کیفیت با پیشنهاد اقدام و کنترل ساختار دادهٔ ورودی |
| `timeseries` | تقسیم زمانی با فاصله و ویژگی‌های تأخیری و پنجره‌ای |
| `theme` / `reporting` | تم مشترک، جدول notebook و گزارش HTML |

فهرست کامل، امضای توابع و راهنمای پارامترها در [مرجع API](docs/API.md) موجود است. [راهنمای مهاجرت](docs/MIGRATION.md)، [گزارش بررسی](docs/REVIEW_FA.md) و [مثال اجرایی](examples/walkthrough.py) نیز همراه پروژه‌اند.

## پیش‌پردازش قابل بازاجرا

ابتدا داده را تقسیم کنید. پارامترهای یادگرفتنی را فقط روی آموزش یاد بگیرید و همان state را روی آزمون اجرا کنید.

```python
from NaviLib import cleaning as cl, feature_engineering as fe

train, test = nv.split_data(labeled_df, target="label")
X_train, y_train = train.drop(columns="label"), train["label"]
X_test = test.drop(columns="label")

prepared, states = fe.chain(X_train, [
    (nv.impute_missing, {"method": "median"}),
    (nv.encode_categorical, {"columns": ["city"], "method": "onehot"}),
    (nv.scale_features, {"columns": ["age"]}),
])
prepared_test = nv.apply_state(X_test, states)
nv.describe_states(states)
nv.save_state(states, "outputs/preprocessing.joblib")
```

برای اعتبارسنجی متقاطع، خود پیش‌پردازش باید داخل پایپ‌لاین باشد. `ChainTransformer` ستون هدف را از مراحل غیرنظارتی دور نگه می‌دارد و مراحل حذف‌کنندهٔ ردیف را رد می‌کند. روش‌های آماری خودکار و ویژگی‌های هدف‌محور همچنان نیاز به انتخاب درست تقسیم‌ها دارند؛ کدگذاری هدف/WOE درون خود از دورهای تصادفی استفاده می‌کند و برای وابستگی‌های زمانی یا گروهی تضمین ویژه‌ای ندارد.

```python
from sklearn.linear_model import LogisticRegression

prep = nv.ChainTransformer([
    (nv.impute_missing, {"method": "median"}),
    (nv.encode_categorical, {"columns": ["city"]}),
    (nv.scale_features, {"columns": ["age"]}),
])
pipeline = nv.make_pipeline(prep, LogisticRegression(max_iter=1000))
validation = nv.cross_validate_model(
    pipeline, X_train, y_train, cv=5, n_jobs=1, verbose=False,
)
artifact = nv.train_model(pipeline, X_train, y_train, verbose=False)
predictions = nv.predict_model(artifact, X_test)
nv.save_model(artifact, "outputs/model.joblib")
reloaded_predictions = nv.predict_model(nv.load_model("outputs/model.joblib"), X_test)
```

`cv` باید با تعداد نمونه‌های کوچک‌ترین کلاس سازگار باشد. برای تکرارهای یک فرد/گروه، `groups=` بدهید؛ برای دادهٔ زمانی از splitter زمانی استفاده کنید. خروجی OOF فقط وقتی ساخته می‌شود که هر ردیف دقیقاً یک بار در مجموعهٔ اعتبارسنجی قرار بگیرد. `task="auto"` یک تشخیص ابتکاری است؛ برای رگرسیون با مقادیر صحیح محدود، `task="regression"` را صریح بنویسید.

## نمودار و جدول

```python
nv.set_theme("dark", font_scale=1.1)
figure = nv.eda.plot_distribution(df, columns="age", kind="hist", show=False)
figure.savefig("distribution.png", dpi=200, bbox_inches="tight")

with nv.theme_context("paper", palette=["#0072B2", "#E69F00", "#009E73"]):
    figure = nv.eda.plot_categorical(df, "city", show=False)

styled = nv.style_table(nv.audit_data(df), caption="Quality review")
```

تم‌های `light`، `dark` و `paper` ارائه می‌شوند؛ `palette`، `font_scale` و `rc` قابل تنظیم‌اند. انتخاب تم فقط روی خروجی‌های بعدی NaviLib اثر دارد و `matplotlib.rcParams` را بیرون فراخوانی نمودار تغییر نمی‌دهد. نقشه‌های رنگی با معنای خاص، مانند همبستگی مثبت/منفی، رنگ‌بندی معنایی خود را حفظ می‌کنند. `show=False` شکل را از pyplot می‌بندد اما شیء Figure برای ذخیره در دسترس می‌ماند. آزمون‌های آماری قدیمی از `show_plot` استفاده می‌کنند و هندل شکل‌ها را در کلید `figures` می‌دهند؛ `show_plot=False` ساخت نمودار آن آزمون‌ها را غیرفعال می‌کند.

## قابلیت‌های کیفیت و زمان

```python
schema = nv.infer_schema(reference_df)
changes = nv.validate_schema(incoming_df, schema)
issues = nv.audit_data(incoming_df, target="label")

train, test = nv.temporal_split(events, "timestamp", test_size=.2, gap=2)
lagged = nv.add_lag_features(events, "sales", time="timestamp", group_by="store", lags=[1, 7])
rolling = nv.add_rolling_features(events, "sales", time="timestamp", group_by="store", windows=[7])

adjusted = nv.statistical_tests.adjust_pvalues([.001, .03, .2], method="fdr_bh")
```

`gap` بر حسب تعداد زمان‌های متمایز است؛ lag و window بر حسب تعداد مشاهده هستند. تاریخ‌های رشته‌ای را ابتدا با `pd.to_datetime` تبدیل کنید. زمان تکراری در یک گروه برای ساخت ویژگی رد می‌شود تا ترتیب مبهم نباشد. برای محاسبهٔ ویژگی یک batch، تاریخچهٔ در دسترس آن را همراهش بدهید؛ این توابع آیندهٔ ناشناخته را پیش‌بینی نمی‌کنند. ساختار داده از روی نمونهٔ مرجع استنباط می‌شود و جای قرارداد دامنهٔ شما را نمی‌گیرد. یافته‌های کیفیت پیشنهاد بررسی‌اند، نه دستور حذف خودکار.

## نصب اختیاری و توسعه

```bash
python -m pip install ".[balance]"    # imbalanced-learn
python -m pip install ".[explain]"    # SHAP
python -m pip install ".[io]"         # Excel و Parquet
python -m pip install ".[notebook]"   # جدول‌های pandas Styler
python -m pytest -q
python -m build
python tools/build_docs.py
python examples/walkthrough.py
```

فایل‌های مدل و state با joblib ذخیره می‌شوند؛ فقط فایل مورد اعتماد را بارگذاری کنید. سازگاری میان نسخه‌های مختلف scikit-learn تضمین نمی‌شود. این کتابخانه ابزار کمکی تحلیل و مدل‌سازی است؛ مدل زبانی یا سرویس هوش مصنوعی مولد داخلی ندارد.
