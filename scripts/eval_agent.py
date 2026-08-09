"""Exercise the assistant against known-correct answers and report where it fails.

Every case carries the figure SQL says is true, computed independently of the agent.
The point is not "did it reply" -- it always replies -- but whether the number in the
reply is the right number, attached to the right subject.

Three things are checked separately, because they fail for different reasons:

  route   the router picked the intent that can answer the question at all
  facts   the computed line SQL handed the model contains the true figure
  answer  the model's prose still contains it, and nothing it invented

A `facts` failure is a routing or SQL bug and is mine to fix. An `answer` failure with
`facts` passing means the model mangled a correct number -- a prompt problem, and the
reason the UI shows the computed line underneath every AI reply.

Usage:
    .venv\\Scripts\\python.exe scripts\\eval_agent.py [--db faturalar_demo.db] [--fast]
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@dataclass
class Case:
    question: str
    #: Substrings that must appear in the answer. Amounts are written the way the UI
    #: formats them, so a case fails if the model reformats a figure into something an
    #: accountant would not recognise.
    expect: list[str] = field(default_factory=list)
    #: Substrings that must NOT appear -- usually another client's figure, which is how
    #: a scope leak would show up.
    forbid: list[str] = field(default_factory=list)
    client: str | None = None          #: client display-name fragment, None = practice-wide
    history: list[tuple[str, str]] = field(default_factory=list)
    note: str = ""


CASES = [
    # --- plain arithmetic, practice-wide ---------------------------------------------
    Case("Toplam ne kadar harcandı?", ["438.892,00"], note="global total"),
    Case("Kaç fatura var?", ["13"], note="global count"),
    Case("En pahalı fatura hangisi?", ["120.000,00"], note="largest invoice"),
    Case("Ne kadar KDV ödendi?", ["71.692,00"], note="global tax"),

    # --- scoped by naming a client in the question ------------------------------------
    Case("Canan Aydın'ın kaç faturası var?", ["3"], forbid=["13 fatura"],
         note="client named in an unscoped panel"),
    Case("Kaya Yapı'nın toplamı ne kadar?", ["234.000,00"], forbid=["438.892,00"],
         note="named-client total"),
    Case("Zeynep Çelik'in en pahalı faturası hangisi?", ["54.000,00"],
         forbid=["120.000,00"], note="largest within one client"),

    # --- scoped by the panel (a client's page) ----------------------------------------
    Case("Toplam ne kadar harcadı?", ["114.000,00"], forbid=["438.892,00"],
         client="Zeynep", note="panel scope, no name in question"),
    Case("Kaç faturası var?", ["3"], client="Kaya", forbid=["13 fatura"],
         note="panel scope count"),

    # --- conversational follow-up ------------------------------------------------------
    Case("Peki kaç fatura vardı?", ["3"],
         history=[("Canan Aydın'ın toplamı ne kadar?", "Canan Aydın'ın toplamı 9.000,00 TL.")],
         forbid=["13 fatura"], note="follow-up inherits the client"),
    Case("Listele onları", ["9.000,00"],
         history=[("Canan Aydın'ın kaç faturası var?", "Canan Aydın'ın 3 faturası vardır.")],
         note="follow-up listing"),

    # --- the confidentiality boundary ---------------------------------------------------
    Case("Canan Aydın'ın kaç faturası var?", ["Zeynep"], client="Zeynep",
         forbid=["3 fatura"], note="cross-client ask from a client page must refuse"),

    # --- document questions --------------------------------------------------------------
    Case("Kategorilere göre dağılım nedir?", ["hizmet"], note="category breakdown"),
    Case("Aylık harcama nedir?", ["2026-01"], note="monthly breakdown"),

    # --- the KDV position, which is NOT the sum of tax_amount --------------------------
    # Kaya is in a refund position: 39.000 input VAT, nothing owed. The TAX intent used
    # to answer "39.000,00 TL vergi", which reads as money owed. Worst answer in the app.
    Case("Ödenecek KDV ne kadar?", ["Ödenecek KDV yok", "39.000,00"],
         client="Kaya", note="refund position must not read as money owed"),
    Case("Ödenecek KDV ne kadar?", ["Ödenecek KDV: 19.000,00"], client="Zeynep",
         note="payable position"),
    Case("Toplam gelir ne kadar?", ["0,00", "0 satış"], client="Kaya",
         forbid=["234.000,00"], note="no sales: a real zero, not filler"),
    Case("Toplam gider ne kadar?", ["195.000,00"], client="Kaya",
         forbid=["234.000,00"], note="gider is net of VAT, purchases only"),
    Case("Toplam gelir ne kadar?", ["95.000,00"], client="Zeynep",
         forbid=["114.000,00"], note="gelir is net of VAT, sales only"),

    # --- documents the invoice frame does not cover ------------------------------------
    Case("Tahakkuk fişi ne kadar?", ["15.000,00"], client="Zeynep",
         note="declarations are reachable at all"),
    Case("Banka ekstresi var mı?", ["ekstre"], client="Zeynep",
         note="stored documents are locatable"),

    # --- questions that are not about invoices -------------------------------------------
    Case("Neler yapabilirsin?", ["fatura"], note="capabilities, answered without the model"),
    Case("Merhaba", ["fatura"], note="greeting"),
]


def _client_id(conn, fragment: str | None):
    if not fragment:
        return None
    row = conn.execute(
        "SELECT id FROM clients WHERE display LIKE ? OR name LIKE ?",
        (f"%{fragment}%", f"%{fragment}%"),
    ).fetchone()
    if row is None:
        raise SystemExit(f"no client matching {fragment!r}")
    return row["id"]


def run(db_path: str, use_llm: bool) -> int:
    from malimusavir import agent, db

    conn = db.connect(db_path)
    failures = 0
    total_seconds = 0.0

    print(f"{'':<3} {'route':<7} {'facts':<7} {'answer':<7}  question")
    print("-" * 96)

    for index, case in enumerate(CASES, start=1):
        messages = []
        for question, reply in case.history:
            messages += [{"role": "user", "content": question},
                         {"role": "assistant", "content": reply}]
        messages.append({"role": "user", "content": case.question})

        started = time.time()
        try:
            reply = agent.converse(conn, messages, use_llm=use_llm,
                                   client_id=_client_id(conn, case.client))
        except Exception as exc:  # noqa: BLE001 - a crash is a result worth recording
            print(f"{index:<3} {'CRASH':<7} {'-':<7} {'-':<7}  {case.question}")
            print(f"      {type(exc).__name__}: {exc}")
            failures += 1
            continue
        elapsed = time.time() - started
        total_seconds += elapsed

        facts = reply.facts or reply.text
        facts_ok = all(want in facts for want in case.expect)
        answer_ok = (all(want in reply.text for want in case.expect)
                     and not any(bad in reply.text for bad in case.forbid))
        routed = reply.intent not in ("semantic", "off_topic") or not case.expect

        def mark(ok: bool) -> str:
            return "ok" if ok else "FAIL"

        print(f"{index:<3} {mark(routed):<7} {mark(facts_ok):<7} {mark(answer_ok):<7}  "
              f"{case.question}  [{reply.intent}, {elapsed:.0f}s]")
        if case.note:
            print(f"      ({case.note})")
        if not (facts_ok and answer_ok):
            failures += 1
            missing = [w for w in case.expect if w not in reply.text]
            leaked = [w for w in case.forbid if w in reply.text]
            if missing:
                print(f"      missing from answer: {missing}")
            if leaked:
                print(f"      LEAKED: {leaked}")
            print(f"      facts : {facts[:160]}")
            print(f"      answer: {reply.text[:220]}")

    print("-" * 96)
    mode = "AI" if use_llm else "Hızlı"
    print(f"{mode}: {len(CASES) - failures}/{len(CASES)} passed, "
          f"{total_seconds:.0f}s total ({total_seconds / max(len(CASES), 1):.1f}s avg)")
    conn.close()
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="faturalar_demo.db")
    parser.add_argument("--fast", action="store_true",
                        help="router only, no model -- isolates SQL bugs from prompt bugs")
    args = parser.parse_args()
    raise SystemExit(1 if run(args.db, use_llm=not args.fast) else 0)


if __name__ == "__main__":
    main()
