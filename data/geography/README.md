# India geographic reference — sources, dates, limitations
Updated: 2026-10-28

Files:
- india_states_uts.csv: 28 states + 8 UTs (reference centres, OpenStreetMap Nominatim approx, 2026-10-28)
- india_districts.csv: district centres where coordinates available (~50, OpenStreetMap approx / project registry, 2026-10-28)
- districts_north/west/up_bihar/east/central/ka_ap_tg/tn_kl/ne/islands.csv: comprehensive district NAME reference (all 36 states/UTs, Census of India / LGD administrative names, 2024-2026). lat/lon intentionally None — no fake coordinates. Search works on names; assessment needs offline centres or live Photon/Nominatim geocoding.
- india_localities.csv: major cities/towns + verified Vellore seed (OpenStreetMap Nominatim)
- ../external/stations.csv: monitoring registry (10 records, CPCB/OpenAQ)
- ../external/vellore_places.csv: verified Vellore seed (OpenStreetMap Nominatim)

Licensing: administrative names are factual; coordinates are approx reference centres
for lookup only, NOT boundaries or measurements. Pollution/model availability decided downstream.
No district boundary GeoJSON is bundled; map uses marker points + live OSM tiles.

