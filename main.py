#!/usr/bin/env python3
import argparse
import csv
import os
import datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom
from collections import deque

# =========================
# IMPORT USER SETTINGS AND STOCK SPLITS
# =========================
from user_settings import (
    TAX_YEAR, PERIOD_START, PERIOD_END,
    TAX_NUMBER, FULL_NAME, ADDRESS, CITY, POST_NUMBER, BIRTH_DATE, EMAIL, PHONE,
    SPLITS,
)

# =========================
# SCRIPT SETTINGS
# =========================
INPUT_FOLDER = "input"
RATE_FOLDER = "rate"
OUTPUT_FOLDER = "output"
OUTPUT_FILENAME = "output.xml"

# Supported actions (Trading 212 export)
SUPPORTED_ACTIONS = {"Market sell", "Market buy", "Limit sell", "Limit buy", "Stop sell"}

# We treat these as sells (disposals)
SELL_ACTIONS = {"Market sell", "Limit sell", "Stop sell"}

# eDavki supports up to 8 decimals for these fields (XSD patterns)
DECIMAL_RULES = {
    "typeDecimalPos14_8": {"int_digits": 14, "scale": 8, "allow_negative": False},
    "typeDecimalPos12_8": {"int_digits": 12, "scale": 8, "allow_negative": False},
    "typeDecimalNeg12_8": {"int_digits": 12, "scale": 8, "allow_negative": True},
}

Q8 = Decimal("0.00000001")


# =========================
# HELPERS
# =========================
def get_files(folder: str) -> list[str]:
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder '{folder}' does not exist.")
    return [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]


def save_file(data: str, output_folder: str = OUTPUT_FOLDER, filename: str = OUTPUT_FILENAME) -> str:
    os.makedirs(output_folder, exist_ok=True)
    path = os.path.join(output_folder, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(data)
    return path


def prettify(elem) -> str:
    rough_string = tostring(elem, "utf-8")
    parsed = minidom.parseString(rough_string)
    return parsed.toprettyxml(indent="  ")


def to_decimal(value) -> Decimal:
    try:
        return Decimal(str(value).strip())
    except (InvalidOperation, ValueError) as e:
        raise ValueError(f"Invalid number: {value}. Details: {e}")


def quantize_8(d: Decimal) -> Decimal:
    return d.quantize(Q8, rounding=ROUND_HALF_UP)


def fmt_decimal(value, xsd_type: str) -> str:
    if xsd_type not in DECIMAL_RULES:
        raise ValueError(f"Unknown XSD decimal type: {xsd_type}")

    rule = DECIMAL_RULES[xsd_type]
    d = to_decimal(value)

    if not rule["allow_negative"] and d < 0:
        raise ValueError(f"Negative value not allowed for {xsd_type}: {value}")

    scale = rule["scale"]
    q = Decimal("1").scaleb(-scale)  # 10^-scale
    d = d.quantize(q, rounding=ROUND_HALF_UP)

    if d == 0:
        d = Decimal("0").quantize(q)

    s = format(d, "f")

    if "." in s:
        s = s.rstrip("0").rstrip(".")

    int_part = s.split(".", 1)[0].lstrip("-")
    if len(int_part) > rule["int_digits"]:
        raise ValueError(f"Too many digits before decimal for {xsd_type}: {s}")

    return s


def parse_date(time_value: str) -> str:
    return time_value.split()[0]


def in_tax_year(date_yyyy_mm_dd: str) -> bool:
    return date_yyyy_mm_dd.startswith(TAX_YEAR + "-")


# =========================
# CSV NORMALIZATION (MULTI-YEAR HEADERS)
# =========================
def find_col(header: list[str], name: str) -> int | None:
    try:
        return header.index(name)
    except ValueError:
        return None


def find_col_startswith(header: list[str], prefix: str) -> int | None:
    for i, h in enumerate(header):
        if h.startswith(prefix):
            return i
    return None


def build_row_mapper(header: list[str]) -> dict[str, int]:
    m: dict[str, int] = {}

    m["Action"] = find_col(header, "Action")
    m["Time"] = find_col(header, "Time")
    m["Ticker"] = find_col(header, "Ticker")
    m["Shares"] = find_col(header, "No. of shares")
    m["Price"] = find_col(header, "Price / share")
    m["PriceCcy"] = find_col(header, "Currency (Price / share)")
    m["FxRate"] = find_col(header, "Exchange rate")

    m["Result"] = find_col(header, "Result")
    m["Total"] = find_col(header, "Total")

    m["ResultCcy"] = find_col(header, "Currency (Result)")
    m["TotalCcy"] = find_col(header, "Currency (Total)")

    required = ["Action", "Time", "Ticker", "Shares", "Price", "PriceCcy", "FxRate"]
    for k in required:
        if m.get(k) is None:
            raise ValueError(f"CSV header missing required column '{k}'")

    if m.get("Result") is None and m.get("Total") is None:
        m["Result"] = find_col_startswith(header, "Result")
        m["Total"] = find_col_startswith(header, "Total")
        if m.get("Result") is None and m.get("Total") is None:
            raise ValueError("CSV header must contain Result or Total column")

    return m


def normalize_row(raw: list[str], m: dict[str, int]) -> dict:
    def get(key: str) -> str:
        idx = m.get(key)
        if idx is None or idx >= len(raw):
            return ""
        return raw[idx].strip()

    base_ccy = get("ResultCcy") or get("TotalCcy") or "EUR"

    return {
        "action": get("Action"),
        "time": get("Time"),
        "ticker": get("Ticker"),
        "shares": get("Shares"),
        "price": get("Price"),
        "price_ccy": get("PriceCcy"),
        "fx_rate": get("FxRate"),
        "base_ccy": base_ccy,
        "result": get("Result"),
        "total": get("Total"),
    }


def read_input_file(filename: str, input_folder: str, state: dict) -> None:
    path = os.path.join(input_folder, filename)
    with open(path, "r", newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        header_row = next(reader)
        mapper = build_row_mapper(header_row)

        for raw in reader:
            if not raw:
                continue
            action = raw[mapper["Action"]].strip()
            if action not in SUPPORTED_ACTIONS:
                continue
            state["rows"].append(normalize_row(raw, mapper))


def load_input_files(input_folder: str, state: dict) -> None:
    input_files = [f for f in get_files(input_folder) if f.lower().endswith(".csv")]
    if not input_files:
        raise FileNotFoundError(f"No CSV files found in {input_folder} folder.")
    for filename in sorted(input_files):
        print(f"Parsing file: {filename}")
        read_input_file(filename, input_folder, state)


# =========================
# FX RATES
# =========================
def read_rate_file(filename: str, rate_folder: str, usd_eur: dict) -> None:
    path = os.path.join(rate_folder, filename)
    with open(path, "r", newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)  # skip header
        for row in reader:
            if row and len(row) >= 2:
                usd_eur[row[0]] = row[1]


def load_usd_eur_rates(rate_folder: str, state: dict) -> None:
    usd_eur = state["usd_eur"]
    rate_files = [f for f in get_files(rate_folder) if f.lower().endswith(".csv")]
    if not rate_files:
        raise FileNotFoundError(f"No exchange rate CSV files found in {rate_folder} folder.")
    for filename in sorted(rate_files):
        read_rate_file(filename, rate_folder, usd_eur)


def find_usd_eur_rate(date: str, usd_eur: dict) -> Decimal:
    dt = date
    while dt not in usd_eur:
        dt_dt = datetime.datetime.strptime(dt, "%Y-%m-%d") - datetime.timedelta(days=1)
        dt = dt_dt.strftime("%Y-%m-%d")
    return to_decimal(usd_eur[dt])


def convert_to_base(price, rate) -> Decimal:
    p = to_decimal(price)
    r = to_decimal(rate)
    if r == 0:
        raise ValueError(f"Invalid exchange rate: {rate}")
    return p / r


def convert_usd_to_eur(price_usd: Decimal, date: str, usd_eur: dict) -> Decimal:
    rate = find_usd_eur_rate(date, usd_eur)
    return price_usd * rate


def compute_eur_unit_price(row: dict, state: dict) -> Decimal:
    date = parse_date(row["time"])
    price = row["price"]
    currency = row["price_ccy"]
    rate = row["fx_rate"]
    base_currency = row["base_ccy"]
    usd_eur = state["usd_eur"]

    if currency == "EUR":
        return to_decimal(price)

    if base_currency == "EUR" and currency == "USD":
        return convert_usd_to_eur(to_decimal(price), date, usd_eur)

    if base_currency == "EUR":
        return convert_to_base(price, rate)

    if base_currency == "USD":
        usd = convert_to_base(price, rate)
        return convert_usd_to_eur(usd, date, usd_eur)

    raise ValueError(f"Unsupported base currency: {base_currency}")


# =========================
# SPLITS
# =========================
def apply_splits_if_needed(ticker: str, current_date: str, fifo_queue, applied_splits: set):
    """
    When we move past a split effective date, adjust all open FIFO lots:
      qty *= ratio
      price_eur /= ratio
    """
    if ticker not in SPLITS:
        return

    for eff_date, ratio in SPLITS[ticker]:
        key = (eff_date, str(ratio))
        if key in applied_splits:
            continue
        if current_date >= eff_date:
            for lot in fifo_queue:
                lot["qty"] = lot["qty"] * ratio
                lot["price_eur"] = lot["price_eur"] / ratio
            applied_splits.add(key)


# =========================
# FIFO MATCHING (ACROSS HISTORY, OUTPUT ONLY TAX_YEAR SELLS + USED BUYS)
# =========================
def fifo_match_for_year(state: dict) -> dict[str, dict]:
    rows_by_ticker: dict[str, list[dict]] = {}
    for r in state["rows"]:
        if r["action"] not in SUPPORTED_ACTIONS:
            continue
        rows_by_ticker.setdefault(r["ticker"], []).append(r)

    for t in rows_by_ticker:
        rows_by_ticker[t].sort(key=lambda x: x["time"])

    out: dict[str, dict] = {}

    for ticker, txs in rows_by_ticker.items():
        fifo = deque()
        applied_splits = set()

        purchases_used_for_year: list[dict] = []
        sales_in_year: list[dict] = []

        for tx in txs:
            action_full = tx["action"]
            action = action_full.split()[1].lower()
            date = parse_date(tx["time"])

            # apply any split that becomes effective by this date
            apply_splits_if_needed(ticker, date, fifo, applied_splits)

            qty = to_decimal(tx["shares"])
            price_eur = compute_eur_unit_price(tx, state)

            if action == "buy":
                fifo.append({"date": date, "qty": qty, "price_eur": price_eur})
                continue

            if action != "sell":
                continue

            remaining = qty
            record = in_tax_year(date)
            if record:
                sales_in_year.append({"date": date, "qty": qty, "price_eur": price_eur})

            while remaining > 0:
                if not fifo:
                    raise ValueError(f"FIFO error: not enough buys to cover sell for {ticker} on {date}")

                lot = fifo[0]
                take = lot["qty"] if lot["qty"] <= remaining else remaining

                if record:
                    purchases_used_for_year.append(
                        {"date": lot["date"], "qty": take, "price_eur": lot["price_eur"]}
                    )

                lot["qty"] -= take
                remaining -= take

                if lot["qty"] <= 0:
                    fifo.popleft()

        if sales_in_year:
            out[ticker] = {"purchases": purchases_used_for_year, "sales": sales_in_year}

    return out


# =========================
# XML BUILDING
# =========================
def header_xml(root) -> None:
    header_elem = SubElement(root, "edp:Header")
    taxpayer = SubElement(header_elem, "edp:taxpayer")
    SubElement(taxpayer, "edp:taxNumber").text = TAX_NUMBER
    SubElement(taxpayer, "edp:taxpayerType").text = "FO"
    SubElement(taxpayer, "edp:name").text = FULL_NAME
    SubElement(taxpayer, "edp:address1").text = ADDRESS
    SubElement(taxpayer, "edp:city").text = CITY
    SubElement(taxpayer, "edp:postNumber").text = POST_NUMBER
    SubElement(taxpayer, "edp:birthDate").text = BIRTH_DATE


def KDVP_metadata(root) -> None:
    kdvp_elem = SubElement(root, "KDVP")
    SubElement(kdvp_elem, "DocumentWorkflowID").text = "O"
    SubElement(kdvp_elem, "Year").text = TAX_YEAR
    SubElement(kdvp_elem, "PeriodStart").text = PERIOD_START
    SubElement(kdvp_elem, "PeriodEnd").text = PERIOD_END
    SubElement(kdvp_elem, "IsResident").text = "true"
    SubElement(kdvp_elem, "TelephoneNumber").text = PHONE
    SubElement(kdvp_elem, "SecurityCount").text = "0"
    SubElement(kdvp_elem, "SecurityShortCount").text = "0"
    SubElement(kdvp_elem, "SecurityWithContractCount").text = "0"
    SubElement(kdvp_elem, "SecurityWithContractShortCount").text = "0"
    SubElement(kdvp_elem, "ShareCount").text = "0"
    SubElement(kdvp_elem, "Email").text = EMAIL


def KVDP_item(root, ticker: str):
    item_elem = SubElement(root, "KDVPItem")
    SubElement(item_elem, "InventoryListType").text = "PLVP"
    SubElement(item_elem, "Name").text = ticker
    SubElement(item_elem, "HasForeignTax").text = "false"
    SubElement(item_elem, "HasLossTransfer").text = "true"
    SubElement(item_elem, "ForeignTransfer").text = "false"
    SubElement(item_elem, "TaxDecreaseConformance").text = "false"

    securities = SubElement(item_elem, "Securities")
    SubElement(securities, "Code").text = ticker
    SubElement(securities, "IsFond").text = "false"

    row_elem = SubElement(securities, "Row")
    SubElement(row_elem, "ID").text = "0"
    return row_elem


def sale(root, date: str, quantity, price) -> None:
    sale_elem = SubElement(root, "Sale")
    SubElement(sale_elem, "F6").text = date
    SubElement(sale_elem, "F7").text = fmt_decimal(quantity, "typeDecimalPos12_8")
    SubElement(sale_elem, "F9").text = fmt_decimal(price, "typeDecimalPos14_8")
    SubElement(sale_elem, "F10").text = "true"


def purchase(root, date: str, quantity, price) -> None:
    purchase_elem = SubElement(root, "Purchase")
    SubElement(purchase_elem, "F1").text = date
    SubElement(purchase_elem, "F2").text = "B"
    SubElement(purchase_elem, "F3").text = fmt_decimal(quantity, "typeDecimalPos12_8")
    SubElement(purchase_elem, "F4").text = fmt_decimal(price, "typeDecimalPos14_8")


def process_transactions(state: dict):
    ns = {
        "xmlns": "http://edavki.durs.si/Documents/Schemas/Doh_KDVP_9.xsd",
        "xmlns:edp": "http://edavki.durs.si/Documents/Schemas/EDP-Common-1.xsd",
    }

    envelope = Element("Envelope", ns)
    header_xml(envelope)
    SubElement(envelope, "edp:AttachmentList")
    SubElement(envelope, "edp:Signatures")

    body = SubElement(envelope, "body")
    SubElement(body, "edp:bodyContent")

    doh = SubElement(body, "Doh_KDVP")
    KDVP_metadata(doh)

    fifo_out = fifo_match_for_year(state)
    tickers = sorted(fifo_out.keys())
    print(f"Tickers with sale in {TAX_YEAR}:", ", ".join(tickers) if tickers else "/")

    purchase_count = 0
    sale_count = 0

    for ticker in tickers:
        row_elem = KVDP_item(doh, ticker)

        for p in fifo_out[ticker]["purchases"]:
            purchase(row_elem, p["date"], p["qty"], p["price_eur"])
            purchase_count += 1

        for s in fifo_out[ticker]["sales"]:
            sale(row_elem, s["date"], s["qty"], s["price_eur"])
            sale_count += 1

        SubElement(row_elem, "F8").text = fmt_decimal("0", "typeDecimalNeg12_8")

    return envelope, purchase_count, sale_count


# =========================
# CLI
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="Convert Trading212 CSV exports into eDavki Doh_KDVP XML (FIFO by year).")
    return parser.parse_args()


def main():
    _args = parse_args()

    state = {
        "usd_eur": {},
        "rows": [],
    }

    load_input_files(INPUT_FOLDER, state)
    load_usd_eur_rates(RATE_FOLDER, state)

    base_set = sorted({r.get("base_ccy", "EUR") for r in state["rows"]})
    print("Base currencies found:", ", ".join(base_set) if base_set else "EUR")

    envelope, purchase_count, sale_count = process_transactions(state)
    xml_output = prettify(envelope)
    output_path = save_file(xml_output, OUTPUT_FOLDER, OUTPUT_FILENAME)

    print("Purchases matched (FIFO) in output:", purchase_count)
    print("Sales in tax year in output:", sale_count)
    print("XML saved to:", output_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print("Error:", err)
