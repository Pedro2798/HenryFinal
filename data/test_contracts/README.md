# Contratos de prueba (sintéticos)

**Tres pares** de contratos (original + enmienda) usados para la demo en vivo.
Son imágenes de documentos renderizadas de forma **reproducible** (ver
`generate_test_contracts.py` para los limpios y `generate_degraded_contracts.py`
para el par con artefactos), pensadas para ejercitar todo el pipeline
`visión → contextualización → extracción` de forma determinista.

| # | Archivo | Rol |
|---|---------|-----|
| 1 | `01_service_agreement_original.png` | Contrato de servicios — **original** |
| 2 | `02_service_agreement_amendment.png` | Contrato de servicios — **enmienda** |
| 3 | `03_nda_original.png` | Acuerdo de confidencialidad (NDA) — **original** |
| 4 | `04_nda_amendment.png` | Acuerdo de confidencialidad (NDA) — **enmienda** |
| 5 | `05_service_agreement_original_dirty.png` | Servicios con artefactos — **original** |
| 6 | `06_service_agreement_amendment_dirty.png` | Servicios con artefactos — **enmienda** |

---

## Par 1 — Caso SIMPLE (Service Agreement)

La enmienda introduce **2 modificaciones** y ningún agregado/eliminación:

| Cláusula | Cambio | Antes | Después |
|----------|--------|-------|---------|
| Clause 3 — Fees | MODIFICACIÓN | USD 5,000 / mes | USD 6,500 / mes (desde 1 Jan 2026) |
| Clause 2 — Term | MODIFICACIÓN | Vence 31 Dec 2025 | Vence 30 Jun 2026 |

**Salida esperada (aprox.):**
- `sections_changed`: `["Clause 3 - Fees", "Clause 2 - Term"]`
- `topics_touched`: `["Pricing", "Term & Termination"]`
- `summary_of_the_change`: dos MODIFICACIONES con sus valores antes/después.

**Ejecutar:**
```bash
python -m src.main data/test_contracts/01_service_agreement_original.png \
                   data/test_contracts/02_service_agreement_amendment.png
```

---

## Par 2 — Caso COMPLEJO (NDA)

La enmienda ejerce los **tres tipos de cambio** a la vez:

| Cláusula | Tipo | Detalle |
|----------|------|---------|
| Clause 4 — Territorial Scope | MODIFICACIÓN | De "Argentina" a "Argentina, Brazil, Chile, Uruguay and the European Union" |
| Clause 3 — Permitted Use | ELIMINACIÓN | Se borra la restricción de ingeniería inversa / decompilación |
| Clause 7 — Data Protection | ADICIÓN | Nueva cláusula GDPR + notificación de brechas en 48 h |

**Salida esperada (aprox.):**
- `sections_changed`: `["Clause 4 - Territorial Scope", "Clause 3 - Permitted Use", "Clause 7 - Data Protection"]`
- `topics_touched`: `["Territorial Scope", "Intellectual Property / Use Restrictions", "Data Protection"]`
- `summary_of_the_change`: una MODIFICACIÓN, una ELIMINACIÓN y una ADICIÓN, con detalles.

**Ejecutar:**
```bash
python -m src.main data/test_contracts/03_nda_original.png \
                   data/test_contracts/04_nda_amendment.png
```

---

## Par 3 — Caso DEGRADADO (robustez de visión)

Mismo contrato que el Par 1, pero con artefactos que simulan una foto
real de un documento mal escaneado:

- Rotación leve (~3°).
- 2 manchas tipo café semi-transparentes — una incluso cubre parcialmente
  la fecha **"30 June 2026"** en la enmienda.
- Sombra de pliegue horizontal.
- Leve desenfoque gaussiano + recompresión JPEG (calidad 72).

**Resultado verificado con `gpt-4o`:** el pipeline igualmente extrae las dos
modificaciones correctas — GPT-4o lee "30 June 2026" *a través* de la mancha.
El prompt de visión instruye emitir `[ILLEGIBLE]` antes que adivinar, así
que en degradaciones extremas el sistema falla con seguridad en vez de
alucinar valores críticos.

**Ejecutar:**
```bash
python -m src.main data/test_contracts/05_service_agreement_original_dirty.png \
                   data/test_contracts/06_service_agreement_amendment_dirty.png
```

---

## Regenerar las imágenes

```bash
pip install pillow==11.0.0
python data/test_contracts/generate_test_contracts.py            # pares 1 y 2 (limpios)
python data/test_contracts/generate_degraded_contracts.py        # par 3 (degradado)
```

> Podés reemplazar estos PNG por escaneos reales o por los documentos del
> Drive de la consigna: el pipeline acepta cualquier `.png/.jpg/.jpeg/.webp/.gif`.
