# deli-mapa-patrones

Mapa interactivo 2D de tickets de soporte coloreados por **patrón**, con **K-NN**
para estimar a qué patrón corresponde un punto que se marca en el gráfico.

La salida es **un único HTML autocontenido**: sin servidor, sin CDN, sin internet.
Se abre con doble clic y se puede compartir por mail.

> Este repo **no incluye datos**. La demo (`salida/demo.html`) se genera con datos
> sintéticos. Para usarlo con datos reales hay que apuntarlo a una base propia con
> el esquema que se describe abajo.

## Demo

```powershell
py -m pip install -r requirements.txt
py -u mapa_patrones.py --demo      # genera salida/demo.html (2.500 tickets falsos, 18 patrones)
```

No necesita base de datos ni credenciales. El aspecto del demo se ajusta con
`--demo-mezcla` (fracción de tickets a mitad de camino entre dos patrones → solapamiento),
`--demo-ruido` (dispersión interna) y `--demo-cruzados` (etiquetas mal asignadas).

## Qué hace

- Proyecta embeddings de tickets (768 dims) a 2D: **PCA 50 → t-SNE** (métrica coseno).
- Dibuja cada ticket como un punto, coloreable por **patrón / categoría / producto / prioridad**.
- Panel de filtros facetados: **categoría, subcategoría, producto, motivo, prioridad,
  tipo de gestión y patrón** (con `seleccionar todo`, `ninguno` y `solo` por valor).
- **Click en el mapa** → coloca un punto y calcula los **K vecinos** (K 1–60) y, con un
  **umbral de similitud** (0–1), decide **qué patrón debería ser**.
- Grilla y ejes en unidades t-SNE, zoom, arrastre y `zoom al punto`.

## Cómo decide el patrón (K-NN)

1. Distancia euclídea en el mapa a cada punto visible (respeta los filtros).
2. Se toman los K más cercanos.
3. **Umbral**: similitud relativa entre el 1.º (1,0) y el K-ésimo (0,0);
   `corte = dmin + (1 − umbral) × (dK − dmin)`. Sólo entran los que caen dentro del corte.
   - `umbral 1.0` → sólo el más cercano
   - `umbral 0.8` → el ~20 % más cercano de los K
   - `umbral 0.0` → todos los K
4. Cada vecino vota con su **patrón asignado**, peso `1/rango` (1, 1/2, 1/3…).
   Gana el patrón con mayor suma. El más cercano gana ~83 % de las veces; si otro
   patrón acumula 3 o más de los K restantes, lo supera.
5. Como referencia se muestra además el patrón más cercano por **coseno 768D** de cada
   vecino (lo que diría un matcher puramente vectorial).

El punto 2D **no** se convierte a un vector de 768 dims: t-SNE no tiene inversa.
La posición sólo determina la vecindad; el puntaje sale de los vecinos reales.

## Uso con datos propios

```powershell
Copy-Item .env.example .env      # servidor, base y credenciales
py -u mapa_patrones.py           # consulta la base, calcula y genera salida/mapa_patrones.html
```

| Flag | Default | Qué hace |
|---|---|---|
| `--patrones` | `0` (todos) | limita a los N patrones de más volumen |
| `--tickets-por-patron` | `55` | tope de tickets por patrón (0 = sin tope) |
| `--desde` | `400000` | id de ticket mínimo a muestrear |
| `--refresh` | — | rehace muestra + PCA/t-SNE |
| `--offline` | — | no toca la base; reusa `datos/base.npz` |
| `--demo` | — | datos sintéticos, sin base |

## Esquema esperado

| Tabla | Columnas usadas |
|---|---|
| `DS_Patron` | `Id, Estado, FrecuenciaUso, Problema, Categoria, Subcategoria, Producto, Motivo, Prioridad, TipoGestion, EmbeddingId` |
| `DS_Patron_Embedding` | `Id, PatronId, Vector` (768 dims) |
| `DS_Patron_Ticket` | `PatronId, TicketId` |
| `DS_Ticket_Embedding` | `TicketId, Tipo, Embedding` (768 dims, `Tipo='contexto'`) |

`Vector`/`Embedding` se leen como JSON (texto). Todo es **sólo lectura**.

## Archivos

```
mapa_patrones.py     generador: consulta, PCA + t-SNE, arma el HTML
conexion.py          conexión pyodbc leyendo .env
datos/               cache local: base.npz + base_meta.json   [gitignored]
salida/              mapa_patrones.html (o demo.html)
```

## Límites conocidos

- La muestra por defecto es de 55 tickets por patrón: subir `--tickets-por-patron`
  o `--desde 0` para más densidad (el t-SNE crece en tiempo).
- La vista 2D preserva **vecindad**, no distancia. Con muchos patrones solapados el
  silhouette 2D baja aunque la proyección sea buena: mirar `trustworthiness`
  (en el panel «Calidad del mapa»).
- Los filtros facetados son **atributos del patrón**, no del ticket.
