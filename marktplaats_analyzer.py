from __future__ import annotations



import json
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, List, Tuple


from datetime import datetime
import math
import numpy as np
from scipy import stats

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS & FOLDERS
# ──────────────────────────────────────────────────────────────────────────────



DATA_FOLDER =  # <-- Correct location of .jsonl files
OUTPUT_FOLDER =            # <-- Where maps should go


MIN_LAT, MAX_LAT = 49.0, 55.0  # NL + BE
MIN_LON, MAX_LON = 2.5, 8.0

STATS_FILE = OUTPUT_FOLDER

# ──────────────────────────────────────────────────────────────────────────────
# ORIGINAL FUNCTIONS – *unchanged*
# ──────────────────────────────────────────────────────────────────────────────

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt



def load_coordinates(jsonl_path: str | Path) -> List[Tuple[float, float]]:
    """Return a list of (lat, lon) tuples from *jsonl_path*, filtered to Europe region."""
    coords: List[Tuple[float, float]] = []

    with open(jsonl_path, "r", encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                location = obj.get("location") or {}
                lat = location.get("latitude")
                lon = location.get("longitude")

                if lat is None or lon is None:
                    continue

                lat = float(lat)
                lon = float(lon)

                # ❌ Filter out coordinates outside Europe/NL
                if not (MIN_LAT <= lat <= MAX_LAT and MIN_LON <= lon <= MAX_LON):
                    print(f"🔎 Skipping outlier at line {line_no}: lat={lat}, lon={lon}")
                    continue

                coords.append((lat, lon))
            except Exception as exc:
                print(f"[line {line_no}] Skipped – could not parse JSON: {exc}")
    return coords


def make_plot(coords: List[Tuple[float, float]], output: str | Path | None) -> None:
    """Create the scatter‑map of *coords* and save/show it."""
    if not coords:
        raise SystemExit("No usable coordinates found – nothing to plot.")

    lats, lons = zip(*coords)

    # Compute dynamic extent with a small margin around the extrema
    margin_deg = 0.2
    extent = [min(lons) - margin_deg, max(lons) + margin_deg,
              min(lats) - margin_deg, max(lats) + margin_deg]

    fig = plt.figure(figsize=(8, 10))
    ax = plt.axes(projection=ccrs.Mercator())
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="aliceblue")
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.6)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6)
    ax.add_feature(cfeature.LAKES.with_scale("50m"), edgecolor="black",
                   facecolor="aliceblue", linewidth=0.4)

    ax.scatter(lons, lats, s=10, alpha=0.8, color="red",
               transform=ccrs.PlateCarree(), zorder=5)

    filename = Path(output).stem.replace("_map", "")
    parts = filename.split("-")

    if len(parts) >= 4:
        title_str = f"{parts[0].capitalize()} Map – {'-'.join(parts[1:4])}"
    else:
        title_str = filename.replace("-", " ").title()

    ax.set_title(title_str, fontsize=14, weight="bold", pad=20)

    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches="tight")
        print(f"📍 Saved map → {output.resolve()}")
        plt.close(fig)
    else:
        print("📍 Showing map interactively")
        plt.show()



# ──────────────────────────────────────────────────────────────────────────────
# NEW ANALYTICS HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def parse_jsonl(path: Path) -> List[dict[str, Any]]:
    """Read an entire .jsonl file → list[dict]."""
    out: List[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception as exc:
                print(f"[{path.name}:{ln}] bad JSON – skipped: {exc}")
    return out


def in_bounds(listing: dict[str, Any]) -> bool:
    loc = listing.get("location") or {}
    lat = loc.get("latitude")
    lon = loc.get("longitude")
    if lat is None or lon is None:
        return False
    try:
        lat = float(lat)
        lon = float(lon)
    except ValueError:
        return False
    return MIN_LAT <= lat <= MAX_LAT and MIN_LON <= lon <= MAX_LON


def _get_attr(listing: dict[str, Any], key: str) -> str | None:
    key = key.strip().lower()
    for section in ("attributes", "extendedAttributes"):
        for attr in listing.get(section, []):
            if attr.get("key", "").strip().lower() == key:
                # Try multiple fallback formats
                val = attr.get("value")
                if isinstance(val, str) and val.strip():
                    return val.strip()
                elif isinstance(val, list) and val:
                    return str(val[0]).strip()
                elif isinstance(attr.get("values"), list) and attr["values"]:
                    return str(attr["values"][0]).strip()
    return None

def descr_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    a = np.asarray(values, dtype=float)
    mean = float(a.mean())
    median = float(np.median(a))
    vmin, vmax = float(a.min()), float(a.max())
    std = float(a.std(ddof=1)) if a.size > 1 else 0.0
    var = float(np.var(a, ddof=1)) if a.size > 1 else 0.0
    cv = float(std / mean) if mean else 0.0
    
    # Compute kurtosis with warnings suppressed
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        kurt = float(stats.kurtosis(a, fisher=True, bias=False)) if a.size > 3 else 0.0
    
    # Replace NaN in kurtosis with None (or 0)
    kurt = replace_nan_with_none(kurt)
    
    return {
        "average": mean,
        "median": median,
        "min": vmin,
        "max": vmax,
        "standard_deviation": std,
        "variance": var,
        "coefficient_of_variation": cv,
        "kurtosis": kurt,
    }


def ratio_priority(listings: list[dict[str, Any]]) -> float:
    dag = sum(1 for l in listings if l.get("priorityProduct") == "DAGTOPPER")
    eligible = sum(1 for l in listings if l.get("priorityProduct") in {"NONE", "DAGTOPPER"})
    return round((dag / eligible) * 100, 2) if eligible else 0.0

def replace_nan_with_none(value):
    """Replace NaN with None."""
    return value if not math.isnan(value) else None


def linreg_by_model(listings: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: defaultdict[str, list[tuple[float, float]]] = defaultdict(list)
    year_buckets: defaultdict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    year_color_buckets: defaultdict[tuple[str, str], Counter] = defaultdict(Counter)
    year_option_buckets: defaultdict[tuple[str, str], Counter] = defaultdict(Counter)

    for l in listings:
        mdl = _get_attr(l, "model")
        year = _get_attr(l, "constructionYear")
        prc_cent = l.get("priceInfo", {}).get("priceCents")
        km = _get_attr(l, "mileage")

        if not (mdl and prc_cent and km):
            continue

        try:
            mileage_val = float(km.replace(".", "").replace(",", ""))
            price_val = float(prc_cent) / 100
            mdl_lower = mdl.lower()
            buckets[mdl_lower].append((mileage_val, price_val))

            if year:
                key = (mdl_lower, year)
                year_buckets[key].append((mileage_val, price_val))

                # Collect color info
                for block in ("attributes", "extendedAttributes"):
                    for attr in l.get(block, []):
                        if attr.get("key", "").strip().lower() in {"color", "interiorcolor"}:
                            color = str(attr.get("value") or (attr.get("values") or [None])[0]).strip().lower()
                            if color:
                                year_color_buckets[key][color] += 1

                # Collect option info
                for block in ("attributes", "extendedAttributes"):
                    for attr in l.get(block, []):
                        if attr.get("key", "").strip().lower() == "options":
                            for option in attr.get("values") or []:
                                if isinstance(option, str):
                                    year_option_buckets[key][option.strip()] += 1

        except ValueError:
            continue

    out: dict[str, Any] = {}

    # Main loop for linear regression by model
    for mdl, pairs in buckets.items():
        if len(pairs) < 3:
            continue
        x_vals = [p[0] for p in pairs]
        y_vals = [p[1] for p in pairs]
        x = np.array(x_vals, dtype=float)
        y = np.array(y_vals, dtype=float)

        # Check if all x values are identical
        if np.all(x == x[0]):
            # If all x values are the same, return 0 for all regression values
            out[mdl] = {
                "r_squared": 0,
                "p_value": 0,
                "slope": 0,
                "intercept": 0,
                "std_err_slope": 0,
                "std_err_intercept": 0,
                "observations": len(pairs),
                "mileage_stats": descr_stats(x_vals),
                "price_stats": descr_stats(y_vals),
                "by_year": {}
            }
            continue

        res = stats.linregress(x, y)

        out[mdl] = {
            "r_squared": replace_nan_with_none(res.rvalue ** 2),
            "p_value": replace_nan_with_none(res.pvalue),
            "slope": replace_nan_with_none(res.slope),
            "intercept": replace_nan_with_none(res.intercept),
            "std_err_slope": replace_nan_with_none(res.stderr),
            "std_err_intercept": replace_nan_with_none(res.intercept_stderr),
            "observations": len(pairs),
            "mileage_stats": descr_stats(x_vals),
            "price_stats": descr_stats(y_vals),
            "by_year": {}
        }

        # Subcategory: by year
        for (mdl_key, year), subpairs in year_buckets.items():
            if mdl_key != mdl or len(subpairs) < 3:
                continue
            x_vals_y = [p[0] for p in subpairs]
            y_vals_y = [p[1] for p in subpairs]
            x_y = np.array(x_vals_y, dtype=float)
            y_y = np.array(y_vals_y, dtype=float)

            # Check if all x values are identical for this subset as well
            if np.all(x_y == x_y[0]):
                out[mdl]["by_year"][year] = {
                    "r_squared": 0,
                    "p_value": 0,
                    "slope": 0,
                    "intercept": 0,
                    "std_err_slope": 0,
                    "std_err_intercept": 0,
                    "observations": len(subpairs),
                    "mileage_stats": descr_stats(x_vals_y),
                    "price_stats": descr_stats(y_vals_y),
                    "color_distribution": {},
                    "top_options": []
                }
                continue

            res_y = stats.linregress(x_y, y_y)

            color_dist = dict(year_color_buckets.get((mdl, year), {}))
            top_options = [opt for opt, _ in year_option_buckets.get((mdl, year), Counter()).most_common(10)]

            out[mdl]["by_year"][year] = {
                "r_squared": replace_nan_with_none(res_y.rvalue ** 2),
                "p_value": replace_nan_with_none(res_y.pvalue),
                "slope": replace_nan_with_none(res_y.slope),
                "intercept": replace_nan_with_none(res_y.intercept),
                "std_err_slope": replace_nan_with_none(res_y.stderr),
                "std_err_intercept": replace_nan_with_none(res_y.intercept_stderr),
                "observations": len(subpairs),
                "mileage_stats": descr_stats(x_vals_y),
                "price_stats": descr_stats(y_vals_y),
                "color_distribution": color_dist,
                "top_options": top_options
            }

    return out



def extract_color_distribution(listings: list[dict[str, Any]]) -> dict[str, int]:
    """Returns a case-insensitive distribution of car colors."""
    from collections import Counter

    color_counter = Counter()
    for l in listings:
        for block in ("attributes", "extendedAttributes"):
            for attr in l.get(block, []):
                if attr.get("key") == "color":
                    val = attr.get("value")
                    if val:
                        color_counter[val.strip().lower()] += 1
    return dict(color_counter)



def brand_descr_stats(listings: list[dict[str, Any]]) -> dict[str, Any]:
    brands: defaultdict[str, list[tuple[float, float, float | None]]] = defaultdict(list)

    for l in listings:
        brand = "cars"
        prc_cent = l.get("priceInfo", {}).get("priceCents")
        km = _get_attr(l, "mileage")
        road_tax_raw = _get_attr(l, "roadTax")

        if not (prc_cent and km):
            continue

        try:
            km_float = float(km.replace(".", "").replace(",", ""))
            price_eur = float(prc_cent) / 100
            road_tax = None
            if road_tax_raw:
                # Extract numeric part, e.g., "221 €/maand" → 221.0
                road_tax = float(road_tax_raw.strip().split()[0].replace(",", "."))
            brands[brand].append((km_float, price_eur, road_tax))
        except ValueError:
            continue

    out: dict[str, Any] = {}
    for br, triplets in brands.items():
        kms, prcs, taxes = zip(*triplets)
        valid_taxes = [t for t in taxes if t is not None]
        out[br] = {
            "mileage_stats": descr_stats(list(kms)),
            "price_stats": descr_stats(list(prcs)),
            "average_road_tax": round(float(np.mean(valid_taxes)), 2) if valid_taxes else None,
            "observations": len(triplets),
        }
    return out

# Plot Data


def histogram_data(values: list[float], bins: int = 10) -> dict[str, list[float]]:
    if not values:
        return {}
    hist, bin_edges = np.histogram(values, bins=bins, density=False)
    density = hist / np.sum(hist)
    return {
        "bin_edges": bin_edges.tolist(),
        "bin_counts": hist.tolist(),
        "bin_density": density.tolist(),
    }



# ──────────────────────────────────────────────────────────────────────────────
# MAIN STATISTICS COMPOSER
# ──────────────────────────────────────────────────────────────────────────────

def build_stats(listings: list[dict[str, Any]], fname: str) -> dict[str, Any]:
    out: dict[str, Any] = {"file_name": fname, "total_listings": len(listings)}


    out["date"] = datetime.now().isoformat()
    # ─── Prices ────────────────────────────────────────────────────────────
    euros = [l["priceInfo"]["priceCents"] / 100 for l in listings if l.get("priceInfo", {}).get("priceCents")]
    out["price_info"] = descr_stats(euros)

    # ─── Seller info ───────────────────────────────────────────────────────
    sellers = Counter()
    verified_ids: set[int] = set()
    unverified_ids: set[int] = set()
    opsticker = Counter()

    for l in listings:
        s = l.get("sellerInformation") or {}
        sellers[s.get("sellerName", "UNKNOWN")] += 1
        if s.get("isVerified"):
            verified_ids.add(s.get("sellerId"))
        else:
            unverified_ids.add(s.get("sellerId"))
        txt = str(l.get("opvalStickerText", "")).lower().strip()
        if txt:
            opsticker[txt] += 1

    out["seller_info"] = {
        "top_10_seller": dict(sellers.most_common(10)),
        "number_verified_sellers": len(verified_ids),
        "number_unverified_sellers": len(unverified_ids),
        "opvalStickerText": dict(opsticker),
        "priority_ratio_dagtopper": ratio_priority(listings),
    }

    # ─── Car stats ─────────────────────────────────────────────────────────
    car: dict[str, Any] = {}

    # construction year distribution
    year_dist = Counter(_get_attr(l, "constructionYear") for l in listings if _get_attr(l, "constructionYear"))
    car["construction_year_distribution"] = dict(year_dist)
    
    
    # Color distribution
    car["color_distribution"] = extract_color_distribution(listings)


    # mileage descriptive stats
    kms = []
    for l in listings:
        km = _get_attr(l, "mileage")
        if km:
            try:
                kms.append(float(km.replace(".", "").replace(",", "")))
            except ValueError:
                pass
    car["mileage_stats"] = descr_stats(kms)

    # seller cities
    city_dist = Counter(
        l.get("location", {}).get("cityName")
        for l in listings
        if l.get("location", {}).get("cityName")
    )

    top_10 = city_dist.most_common(10)
    top_10_keys = set(city for city, _ in top_10)
    other_total = sum(count for city, count in city_dist.items() if city not in top_10_keys)

    # Build final dict
    city_dist_dict = dict(top_10)
    if other_total:
        city_dist_dict["Other"] = other_total

    car["seller_city_distribution"] = city_dist_dict

    # generic attribute distributions
    attr_keys = ("condition", "fuel", "energyLabel", "transmission", "model", "priceType", "imported", "color")
    distributions = {key: Counter() for key in attr_keys}

    for l in listings:
        for k in attr_keys:
            val = _get_attr(l, k)
            if val:
                distributions[k][val.strip().lower()] += 1

    # convert Counters to plain dicts
    for key, dist in distributions.items():
        car[f"{key.lower()}_distribution"] = dict(dist)


    # regression + brand stats
    car["price_on_mileage_regression_by_model"] = linreg_by_model(listings)
    car["brand_descriptive_stats"] = brand_descr_stats(listings)
    
    
    # ─── Precomputed histogram data for future plots ─────────────
    plot_data = {}

    # Overall
    plot_data["overall_price_histogram"] = histogram_data(euros)
    plot_data["overall_mileage_histogram"] = histogram_data(kms)

    # Per-model histograms
    model_price_map = defaultdict(list)
    model_km_map = defaultdict(list)

    for l in listings:
        model = _get_attr(l, "model")
        if not model:
            continue
        try:
            price = l.get("priceInfo", {}).get("priceCents", 0) / 100
            km = float(_get_attr(l, "mileage").replace(".", "").replace(",", ""))
            model_price_map[model.lower()].append(price)
            model_km_map[model.lower()].append(km)
        except:
            continue

    plot_data["price_histograms_by_model"] = {
        mdl: histogram_data(prices)
        for mdl, prices in model_price_map.items() if len(prices) >= 3
    }
    plot_data["mileage_histograms_by_model"] = {
        mdl: histogram_data(kms)
        for mdl, kms in model_km_map.items() if len(kms) >= 3
    }

    car["histograms_for_plots"] = plot_data
    # Average road tax by model
    model_roadtax_map = defaultdict(list)

    for l in listings:
        model = _get_attr(l, "model")
        road_tax_raw = _get_attr(l, "roadTax")
        if not model or not road_tax_raw:
            continue
        try:
            tax = float(road_tax_raw.strip().split()[0].replace(",", "."))
            model_roadtax_map[model.lower()].append(tax)
        except ValueError:
            continue

    car["average_road_tax_by_model"] = {
        model: round(float(np.mean(taxes)), 2)
        for model, taxes in model_roadtax_map.items() if taxes
    }

    out["car_stats"] = car
    return out


# ──────────────────────────────────────────────────────────────────────────────
# UTILITY – append stats to JSON file (compound structure)
# ──────────────────────────────────────────────────────────────────────────────

def append_stats(stats: dict[str, Any], brand: str) -> None:
    output_dir = Path("src/lib/secondhand")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{brand.lower()}.json"

    # If the file already exists, append the new report to a list
    if output_path.exists():
        try:
            with output_path.open("r", encoding="utf-8") as f:
                existing_data = json.load(f)
            if isinstance(existing_data, list):
                existing_data.append(stats)
            else:
                existing_data = [existing_data, stats]
        except Exception:
            existing_data = [stats]
    else:
        existing_data = [stats]

    # Write back to file
    output_path.write_text(json.dumps(existing_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ Appended stats to {output_path.resolve()}")

# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate maps + summary statistics for .jsonl car listings")
    parser.add_argument("files", nargs="*", help="Specific .jsonl files to process (default: all in CWD)")
    args = parser.parse_args()

    files = [Path(f) for f in args.files] if args.files else list(DATA_FOLDER.glob("*.jsonl"))
    if not files:
        print(f"⚠️ No .jsonl files found in {DATA_FOLDER.resolve()}")
        return

    for f in files:
        brand = f.stem.split("_")[0].lower()  # e.g. 'skoda_2025-06-18' → 'skoda'
        print(f"\n📦 Processing {f.name} for brand → {brand}")

        all_listings = parse_jsonl(f)

        # ───── Create and save map image ─────
        coords = load_coordinates(f)
        if coords:
            try:
                map_path = Path("public/maps") / f"{f.stem}_map.png"
                map_path.parent.mkdir(parents=True, exist_ok=True)
                make_plot(coords, map_path)
                print(f"🖼️ Map saved to: {map_path}")
            except Exception as exc:
                print(f"⚠️ Could not create map for {f.name}: {exc}")
        else:
            print(f"⚠️ No valid coordinates in {f.name} – skipping map.")

        # ───── Build stats ─────
        stats = build_stats(all_listings, f.name)

        # ───── Append stats to correct brand file ─────
        json_output_path = Path("") / f"{brand}.json"
        json_output_path.parent.mkdir(parents=True, exist_ok=True)

        existing_data = []
        if json_output_path.exists():
            try:
                with open(json_output_path, "r", encoding="utf-8") as existing_file:
                    existing_data = json.load(existing_file)
                    if not isinstance(existing_data, list):
                        existing_data = []
            except Exception as e:
                print(f"⚠️ Could not load existing data for {brand}: {e}")

        existing_data.append(stats)

        with open(json_output_path, "w", encoding="utf-8") as out:
            json.dump(existing_data, out, indent=2, ensure_ascii=False)

        print(f"✅ Stats appended to {json_output_path.resolve()}")



if __name__ == "__main__":
    main()
