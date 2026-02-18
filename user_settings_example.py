from decimal import Decimal

# =========================
# USER SETTINGS (EDIT THIS)
# =========================
TAX_YEAR = "2024"
PERIOD_START = f"{TAX_YEAR}-01-01"
PERIOD_END = f"{TAX_YEAR}-12-31"

TAX_NUMBER = "12345678"
FULL_NAME = "Full name"
ADDRESS = "Address"
CITY = "City"
POST_NUMBER = "1000"
BIRTH_DATE = "1995-12-31"
EMAIL = "your-email@should-go.here"
PHONE = "069240240"

# =========================
# STOCK SPLITS (MANUAL)
# =========================
# Format: ticker: list of (effective_date, ratio)
SPLITS = {
    "NVDA": [
        ("2021-07-20", Decimal("4")),   # NVDA 4:1 split effective 2021-07-20
        ("2024-06-10", Decimal("10")),  # NVDA 10:1 split effective 2024-06-10
    ],
}
