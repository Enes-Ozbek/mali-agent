"""Mali Musavir -- local Turkish e-Arsiv invoice assistant.

    python main.py --ingest "C:\\faturalar"
    python main.py --ask "Turkcell'e toplam ne kadar odedim"
    python main.py --stats
    python main.py --review
    python main.py --serve
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from malimusavir import db, rag, router, stats
from malimusavir.extractors.base import FIELDS
from malimusavir.pipeline import extract_from_pdf, find_pdfs


def money(value: float | None) -> str:
    """Format in Turkish convention: 1.234,56"""
    if value is None:
        return "-"
    return f"{value:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")


def cmd_ingest(args) -> int:
    if args.dry_run:
        return _dry_run(args)

    conn = db.connect(args.db)
    print(f"Ingesting {args.ingest} ...")
    report = _run_ingest(conn, args)

    print(f"\n{report.inserted} new, {report.updated} updated, {report.skipped} unchanged")
    if report.flagged:
        print(f"\n{len(report.flagged)} invoice(s) need review:")
        for line in report.flagged:
            print(f"  ! {line}")
    if report.failed:
        print(f"\n{len(report.failed)} file(s) failed:")
        for path, reason in report.failed:
            print(f"  x {path}: {reason}")

    if not args.no_embed and report.total:
        print("\nEmbedding for search ...")
        try:
            embedded = rag.embed_pending(conn)
            print(f"  {embedded} invoice(s) embedded")
        except Exception as exc:  # noqa: BLE001 - ingest already succeeded
            print(f"  skipped: {exc}")

    conn.close()
    return 1 if report.failed else 0


def _run_ingest(conn, args):
    from malimusavir.pipeline import ingest_folder

    def progress(path, invoice, result):
        mark = {"inserted": "+", "updated": "~", "skipped": "="}[result.value]
        flag = "  [REVIEW]" if invoice.needs_review else ""
        print(f"  {mark} {path.name[:52]:<52} {money(invoice.total_amount):>12}{flag}")

    return ingest_folder(conn, args.ingest, use_llm=args.llm_category,
                         on_progress=progress, client_name=args.client)


def _dry_run(args) -> int:
    """Extract and print without writing anything."""
    paths = list(find_pdfs(args.ingest))
    if not paths:
        print(f"No PDFs found under {args.ingest}")
        return 1

    for index, path in enumerate(paths, start=1):
        invoice = extract_from_pdf(path, use_llm=args.llm_category)
        status = "REVIEW" if invoice.needs_review else "ok"
        print(f"\n[{index}/{len(paths)}] {path.name} -> {invoice.profile} [{status}]")
        for name in FIELDS:
            value = getattr(invoice, name)
            shown = money(value) if name.endswith("_amount") else value
            print(f"    {name:<15} {str(shown):<48} ({invoice.field_sources.get(name, '-')})")
        if invoice.needs_review:
            print(f"    ! {', '.join(invoice.review_reasons)}")
    print(f"\n{len(paths)} file(s) inspected. Nothing was written (--dry-run).")
    return 0


def cmd_ask(args) -> int:
    conn = db.connect(args.db)
    if db.count(conn) == 0:
        print("No invoices stored yet. Run --ingest first.")
        return 1

    # Arithmetic goes to SQL. Only questions the router does not recognise reach the
    # model, so a figure shown to the user is computed rather than generated.
    if not args.semantic:
        parsed = router.classify(
            args.ask,
            vendors=router.known_vendors(conn),
            categories=router.known_categories(conn),
        )
        if args.explain:
            print(f"[intent={parsed.intent.value}"
                  + (f" trigger={parsed.matched!r}" if parsed.matched else "")
                  + (f" vendors={parsed.vendors!r}" if parsed.vendors else "")
                  + (f" category={parsed.category!r}" if parsed.category else "")
                  + (f" since={parsed.since} until={parsed.until}" if parsed.since else "")
                  + "]\n")
        computed = router.answer(conn, parsed)
        if computed is not None:
            print(computed.text)
            conn.close()
            return 0

    pending = rag.embed_pending(conn)
    if pending:
        print(f"({pending} invoice(s) embedded first)\n")

    reply, hits = rag.answer(conn, args.ask, k=args.top_k)
    print(reply)

    if hits:
        print("\nKaynak faturalar:")
        for hit in hits:
            print(f"  [{hit['score']:.2f}] {hit['date']}  {money(hit['total_amount']):>10} TL  "
                  f"{(hit['vendor'] or '')[:44]}")
    conn.close()
    return 0


def cmd_stats(args) -> int:
    conn = db.connect(args.db)
    frame = stats.load_frame(conn)
    frame = stats.date_range(frame, args.since, args.until)

    if frame.empty:
        print("No invoices in range.")
        return 1

    summary = stats.totals(frame)
    print("=" * 64)
    print(f"{summary.invoices} fatura   {summary.first_date} .. {summary.last_date}")
    print(f"Toplam : {money(summary.total):>14} TL")
    print(f"KDV    : {money(summary.tax):>14} TL")
    print(f"Matrah : {money(summary.net):>14} TL")
    if summary.flagged:
        print(f"({summary.flagged} fatura kontrol bekliyor -- bkz. --review)")
    if summary.mixed_currency:
        # Summing across currencies produces a number in no currency at all.
        print(f"\n!! UYARI: birden fazla para birimi var ({', '.join(summary.currencies)}). "
              f"Yukaridaki toplamlar anlamsizdir.")

    _table("KATEGORIYE GORE", stats.by_category(frame), "category")
    _table("SATICIYA GORE", stats.by_vendor(frame), "vendor")

    print("\nAYLIK")
    for _, row in stats.by_month(frame).iterrows():
        print(f"  {row['ay']}   {money(row['toplam']):>12} TL   ({int(row['adet'])} fatura)")

    print("\nEN BUYUK FATURALAR")
    for _, row in stats.largest(frame).iterrows():
        date = row["date"].date().isoformat() if row["date"] is not None else "-"
        print(f"  {date}  {money(row['total_amount']):>12} TL  "
              f"{(row['vendor'] or '')[:40]}")

    recurring = stats.recurring_vendors(frame)
    if not recurring.empty:
        print("\nDUZENLI (ABONELIK BENZERI)")
        for _, row in recurring.iterrows():
            print(f"  {(row['vendor'] or '')[:40]:<42} {money(row['toplam']):>12} TL  "
                  f"{int(row['adet'])} fatura, ~{row['ortalama_gun']:.0f} gunde bir")
        committed = float(recurring["aylik_ortalama"].sum())
        print(f"  {'-> aylik duzenli gider':<42} {money(committed):>12} TL")

    conn.close()
    return 0


def _table(title: str, frame, key: str) -> None:
    if frame.empty:
        return
    print(f"\n{title}")
    for _, row in frame.iterrows():
        print(f"  {str(row[key])[:40]:<42} {money(row['toplam']):>12} TL  "
              f"({int(row['adet'])} fatura)")


def cmd_review(args) -> int:
    conn = db.connect(args.db)
    rows = db.flagged_invoices(conn)
    if not rows:
        print(f"All {db.count(conn)} invoice(s) look clean.")
        return 0

    print(f"{len(rows)} invoice(s) need review:\n")
    for row in rows:
        print(f"  {row['invoice_no']}  {row['date']}  {money(row['total_amount']):>12} TL  "
              f"{(row['vendor'] or '')[:36]}")
        print(f"      {row['review_reasons']}")
        print(f"      {row['source_path']}")
    conn.close()
    return 0


def cmd_ingest_archive(args) -> int:
    from malimusavir import pipeline

    conn = db.connect(args.db)
    print(f"Ingesting archive {args.ingest_archive} ...")

    def progress(item, client):
        print(f"  {client.label:14} {item.year} {item.doc_type:14} {item.path.name[:44]}")

    report = pipeline.ingest_archive(
        conn, args.ingest_archive, only_client=args.client,
        use_llm=args.llm_category, on_progress=progress,
    )

    inv = report.invoices
    print(f"\n{len(report.clients)} client(s): {', '.join(report.clients) or '-'}")
    print(f"invoices     : {inv.inserted} new, {inv.updated} updated, {inv.skipped} unchanged")
    print(f"beyannameler : {report.declarations} new")
    print(f"diğer belge  : {report.documents} new")

    if report.misfiled:
        print(f"\n{len(report.misfiled)} belge yanlış yıl klasöründe:")
        for line in report.misfiled:
            print(f"  ! {line}")
    if inv.flagged:
        print(f"\n{len(inv.flagged)} fatura inceleme bekliyor:")
        for line in inv.flagged:
            print(f"  ! {line}")
    if report.problems:
        print(f"\n{len(report.problems)} sorun (atlandı):")
        for path, reason in report.problems:
            print(f"  x {Path(path).name}: {reason}")
    if inv.failed:
        print(f"\n{len(inv.failed)} dosya okunamadı:")
        for path, reason in inv.failed:
            print(f"  x {Path(path).name}: {reason}")

    if not args.no_embed and inv.total:
        print("\nEmbedding for search ...")
        try:
            print(f"  {rag.embed_pending(conn)} invoice(s) embedded")
        except Exception as exc:  # noqa: BLE001 - ingest already succeeded
            print(f"  skipped: {exc}")

    conn.close()
    return 1 if (report.problems or inv.failed) else 0


def cmd_clients(args) -> int:
    from malimusavir import clients as clients_mod

    conn = db.connect(args.db)
    rows = clients_mod.summaries(conn)
    unassigned = conn.execute(
        "SELECT COUNT(*) n, COALESCE(SUM(total_amount), 0) t "
        "FROM invoices WHERE client_id IS NULL"
    ).fetchone()

    if not rows and not unassigned["n"]:
        print("Hiç müşteri yok. --ingest-archive ile bir arşiv yükleyin.")
        return 1

    print(f"{'müşteri':22} {'fatura':>7} {'toplam':>14} {'işaretli':>9}  yıllar")
    print("-" * 68)
    for row in rows:
        years = ", ".join(str(y) for y in row["years"]) or "-"
        print(f"{row['label'][:22]:22} {row['invoices']:7} "
              f"{money(row['total']):>14} {row['flagged']:9}  {years}")
    if unassigned["n"]:
        print(f"{'(atanmamış)':22} {unassigned['n']:7} {money(unassigned['t']):>14}")
    conn.close()
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    from malimusavir import api

    api.DB_PATH = args.db
    print(f"Mali Müşavir -- http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
    uvicorn.run(api.app, host="127.0.0.1", port=args.port, log_level="warning")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mali-musavir",
        description="Local Turkish e-Arsiv invoice analysis. Nothing leaves this machine.",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--ingest", metavar="FOLDER", help="Extract and store invoice PDFs")
    action.add_argument("--ask", metavar="SORU", help="Ask a question about your invoices")
    action.add_argument("--stats", action="store_true", help="Show spend aggregates")
    action.add_argument("--review", action="store_true",
                        help="List invoices flagged during extraction")
    action.add_argument("--serve", action="store_true",
                        help="Run the local web dashboard")
    action.add_argument("--ingest-archive", metavar="ROOT",
                        help="Ingest a client archive: <ROOT>/<client>/<year>/<type>/*.pdf")
    action.add_argument("--clients", action="store_true",
                        help="List clients and their totals")

    parser.add_argument("--db", metavar="PATH", default=None, help="Database file")
    parser.add_argument("--client", metavar="NAME", default=None,
                        help="With --ingest: file everything under this client. "
                             "With --ingest-archive: limit to this one client.")
    parser.add_argument("--port", type=int, default=8000,
                        help="With --serve: port to listen on")
    parser.add_argument("--dry-run", action="store_true",
                        help="With --ingest: extract and print, write nothing")
    parser.add_argument("--llm-category", action="store_true",
                        help="Use the model for category when no keyword rule matches "
                             "(slow, and low accuracy on CPU -- results are flagged)")
    parser.add_argument("--no-embed", action="store_true",
                        help="With --ingest: skip building search embeddings")
    parser.add_argument("--top-k", type=int, default=rag.TOP_K,
                        help="With --ask: how many invoices to retrieve")
    parser.add_argument("--semantic", action="store_true",
                        help="With --ask: skip the SQL router and always use search")
    parser.add_argument("--explain", action="store_true",
                        help="With --ask: show how the question was classified")
    parser.add_argument("--since", metavar="YYYY-MM-DD", help="With --stats: start date")
    parser.add_argument("--until", metavar="YYYY-MM-DD", help="With --stats: end date")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # `is not None`, not truthiness: an empty --ask "" or --ingest "" is falsy and
    # would silently fall through to --review instead of reporting a bad argument.
    if args.ingest is not None:
        if not args.ingest.strip():
            print("--ingest needs a folder path")
            return 2
        return cmd_ingest(args)
    if args.ask is not None:
        if not args.ask.strip():
            print("--ask needs a question")
            return 2
        return cmd_ask(args)
    if args.ingest_archive is not None:
        if not args.ingest_archive.strip():
            print("--ingest-archive needs an archive root path")
            return 2
        return cmd_ingest_archive(args)
    if args.stats:
        return cmd_stats(args)
    if args.serve:
        return cmd_serve(args)
    if args.clients:
        return cmd_clients(args)
    return cmd_review(args)


if __name__ == "__main__":
    sys.exit(main())
