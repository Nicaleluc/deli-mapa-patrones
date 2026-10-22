# AGENTS.md — deli-mapa-patrones

Herramienta de **sólo lectura** que genera un mapa 2D de tickets por patrón con K-NN.

## Reglas

1. **Nunca escribir en la base.** Todo el proyecto son `SELECT`. No debe haber
   `INSERT/UPDATE/DELETE`.
2. **No commitear datos.** `salida/mapa_patrones.html` contiene ids y textos reales:
   no se versiona. Sólo se versiona `salida/demo.html`, que sale de
   `py -u mapa_patrones.py --demo` (datos sintéticos).
3. **Credenciales sólo en `.env`** (gitignored). Nunca hardcodear.
4. **`datos/` es caché** (`base.npz` + `base_meta.json`), gitignored. Si se borra,
   hay que correr sin `--offline` para reconstruirlo (requiere base).
5. **No inventar campos**: los atributos salen de `DS_Patron`
   (`Categoria, Subcategoria, Producto, Motivo, Prioridad, TipoGestion`).
6. **Metodología fija**: vista = PCA50 + t-SNE (coseno); decisión = K-NN ponderada por
   rango sobre el **patrón asignado** de cada vecino. El más cercano por coseno es
   sólo referencia.
7. **Consultas acotadas**: `DS_Ticket_Embedding` tiene cientos de miles de filas.
   Consultar por `TicketId IN (...)` (hay índice por `(TicketId, Tipo)`), nunca
   `LIKE` sobre texto ni full scans.

## Verificación

```powershell
py -m py_compile mapa_patrones.py
py -u mapa_patrones.py --demo        # no requiere base; debe imprimir "[html] .../demo.html"
```

Los conteos de `[metricas]` deben coincidir con la cantidad de puntos generados.

## Antes de publicar cambios

```powershell
git status --short                   # no debe aparecer .env ni datos/ ni mapa_patrones.html
```

Si agregás un artefacto generado, verificá que no contenga ids ni textos reales.
