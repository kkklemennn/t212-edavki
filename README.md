# Trading212 CSV to eDavki XML (Doh-KDVP)

Skripta za pretvorbo Trading212 CSV datotek v eDavki XML pripomore k hitrejšemu in bolj organiziranemu ustvarjanju XML datotek za oddajo davčne napovedi.

## Izjava o omejitvi odgovornosti

**OPOZORILO:**  
Ta skripta je zgolj pripomoček, ki poenostavi generiranje XML datoteke za oddajo davčne napovedi. Pred oddajo XML datoteke **obvezno ročno preveri** vse vnose. Z uporabo skripte sprejemaš popolno odgovornost za morebitne napake, izgube ali škodo, ki bi nastale zaradi nepravilno generiranih podatkov. Avtor skripte ne sprejema odgovornosti za kakršnekoli posledice.

## Posodobitve
- ***23.02.2026**
  - Dodana zaznava “wash-sale” situacij po 5. odstavku 97. člena ZDoh-2 (pravilo ±30 dni pri ponovnem nakupu istovrstnega kapitala) – skripta izpiše opozorila v konzolo, izhodni XML ostane nespremenjen.
- **19.02.2026**
  - Optimizacija in izboljšave delovanja skripte s cachingom pretvorbe valut
- **18.02.2026:**
  - Izboljšana podpora za več različnih Trading212 CSV headerjev.
  - Implementiran FIFO obračun čez celotno zgodovino, z izpisom samo prodaj za izbrano TAX_YEAR.
  - Dodana podpora za ročni vnos stock splitov (SPLITS) z avtomatsko prilagoditvijo FIFO zaloge.
  - Ločene uporabniške nastavitve (user_settings_example.py + lokalni user_settings.py).
  - Pretvorba valut sedaj uporablja dnevni tečaj Banke Slovenije (BSI API) za vse podprte valute, namesto lokalne CSV tečajnice.
- **18.02.2026:**
  - Forkano iz [t212-davki](https://github.com/Neophytez/t212-edavki)

## Kako deluje skripta?

1. **Uvoz CSV datotek**  
   Skripta prebere vse CSV datoteke iz mape `input`.

2. **Filtriranje transakcij**  
   Upoštevajo se samo naslednje vrste:
   - **market buy**
   - **market sell**
   - **limit buy**
   - **limit sell**  
   - **stop sell**  
   
   Ostale vrstice (dividende, obresti ipd.) se ignorirajo.

3. **Pretvorba cen v EUR**  
   - Če je osnovna valuta EUR, se cena uporabi neposredno.
   - Če je valuta drugačna (npr. USD, CHF, GBP …), se uporabi dnevni tečaj Banke Slovenije ([BSI API](https://api.bsi.si/exchange/daily)) na dan transakcije.
   
4. **Generiranje XML**  
   - Za vsak ticker, ki ima vsaj eno prodajo, se ustvari KDVPItem.
   - XML je pripravljen za uvoz v eDavki → Doh-KDVP → Uvoz popisnih listov.

## Navodila za uporabo

1. **Namesti Python**  
   - Prenesi [Python](https://www.python.org/downloads/windows/) (sledi [navodilom za namestitev](https://realpython.com/installing-python/))
   - Med namestitvijo obkljukaj *"Add Python to PATH"*

2. **Prenesi repozitorij**  
   - Klikni **Code** → **Download ZIP**
   - Razširi arhiv

3. **Pripravi CSV datoteke**  
   - Iz Trading212 izvozi CSV datoteke (označi vse 4 opcije: Orders, Dividends, Transactions, Interest)
   - Kopiraj CSV datoteke v mapo `t212-edavki-main/input` (skripta podpira več CSV datotek hkrati)

4. **Uredi osebne podatke**
   - Odpri `user_settings_example.py`
   - Vpiši svoje podatke in preimenuj v `user_settings.py`

5. **Zaženi skripto**  
     ```
     python main.py
     ```

6. **Rezultat** 
- Če se izpiše sporočilo:
  ```
  Your XML file is located inside output folder.
  ```
  si uspešno ustvaril XML datoteko, pripravljeno na uvoz v eDavki.
- XML datoteka se ustvari v mapi `output`
- Datoteko uvoziš v eDavki (Doh-KDVP)

## Podpri delo

Če ti je skripta prihranila čas in trud, mi lahko častiš pivo.

[![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://www.paypal.com/donate/?hosted_button_id=4AVZZEVPA7Q58)

# TODO:
- Code review and cleanup
- CLI print organize and cleanup