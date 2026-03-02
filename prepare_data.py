import zipfile
import shutil
import os
import csv
import fitdecode


RECOVER_PATH = os.path.expanduser("~/Drive/Recover")
RECOVER_GADGETBRIDGE_PATH = os.path.join(RECOVER_PATH, "Gadgetbridge.zip")
RECOVER_STT_PATH = os.path.join(
    RECOVER_PATH, "simpletimetracker/stt_records_automatic.csv"
)
DATA_PATH = "./data"
OUTPUT_CSV = os.path.join(DATA_PATH, "daily_health_stats.csv")
GADGETBRIDGE_PATH = os.path.join(DATA_PATH, "Gadgetbridge")
STT_PATH = os.path.join(DATA_PATH, "stt_records_automatic.csv")
DEVICE_PATH = os.path.join(GADGETBRIDGE_PATH, "files/E4:A0:45:B3:99:13")
GARMIN_PATH = os.path.join(DATA_PATH, "Garmin")
PATH_SLEEP = os.path.join(GARMIN_PATH, "SLEEP")
PATH_HRV = os.path.join(GARMIN_PATH, "HRV_STATUS")


def check_data_is_fresh() -> bool:
    if not os.path.exists(RECOVER_STT_PATH) or not os.path.exists(STT_PATH):
        return False

    timestamp1 = os.path.getmtime(RECOVER_STT_PATH)
    timestamp2 = os.path.getmtime(STT_PATH)

    return timestamp1 == timestamp2


def copy_files():
    os.makedirs(DATA_PATH, exist_ok=True)

    if os.path.exists(RECOVER_GADGETBRIDGE_PATH):
        with zipfile.ZipFile(RECOVER_GADGETBRIDGE_PATH, "r") as zip_ref:
            zip_ref.extractall(GADGETBRIDGE_PATH)
    else:
        print("Warning: Gadgetbridge.zip not found")
    if os.path.exists(RECOVER_STT_PATH):
        shutil.copy2(RECOVER_STT_PATH, STT_PATH)
    else:
        print("Warning: stt_records_automatic.csv not found")
    if os.path.exists(DEVICE_PATH):
        if os.path.exists(GARMIN_PATH):
            shutil.rmtree(GARMIN_PATH)
        shutil.move(DEVICE_PATH, GARMIN_PATH)
    else:
        print("Warning: device not found")
    if os.path.exists(GADGETBRIDGE_PATH):
        shutil.rmtree(GADGETBRIDGE_PATH)


def get_minutes(start, end):
    return (end - start).total_seconds() / 60


def parse_hrv_folder(folder_path):
    """Scans HRV folder to map Date -> Overnight HRV (ms)."""
    hrv_map = {}
    if not os.path.exists(folder_path):
        return hrv_map

    for root, dirs, files in os.walk(folder_path):
        for f in files:
            if not f.endswith(".fit"):
                continue

            try:
                with fitdecode.FitReader(os.path.join(root, f)) as fit:
                    for frame in fit:
                        if (
                            isinstance(frame, fitdecode.FitDataMessage)
                            and frame.name == "hrv_status_summary"
                        ):
                            if frame.has_field("weekly_average"):
                                val = frame.get_value("last_night_average")
                                if val is None:
                                    val = frame.get_value("weekly_average")

                                if frame.has_field("timestamp"):
                                    ts = frame.get_value("timestamp")
                                    date_key = ts.strftime("%Y-%m-%d")
                                    hrv_map[date_key] = val
            except:
                continue
    return hrv_map


def parse_sleep_file(filepath):
    """Extracts stages with Tail-Trim logic and Assessment scores."""
    events = []
    file_end_time = None

    stats = {
        "date": None,
        "sleep_score": None,
        "recovery_score": None,
        "deep_m": 0.0,
        "light_m": 0.0,
        "rem_m": 0.0,
        "awake_m": 0.0,
        "total_sleep_m": 0.0,  # Excluding Awake
        "time_in_bed_m": 0.0,  # Including Awake
    }

    try:
        with fitdecode.FitReader(filepath) as fit:
            for frame in fit:
                if isinstance(frame, fitdecode.FitDataMessage):
                    if frame.has_field("timestamp"):
                        file_end_time = frame.get_value("timestamp")

                    if frame.name == "sleep_assessment":
                        if frame.has_field("overall_sleep_score"):
                            stats["sleep_score"] = frame.get_value(
                                "overall_sleep_score"
                            )
                        if frame.has_field("sleep_recovery_score"):
                            stats["recovery_score"] = frame.get_value(
                                "sleep_recovery_score"
                            )
                        if frame.has_field("timestamp"):
                            stats["date"] = frame.get_value("timestamp").strftime(
                                "%Y-%m-%d"
                            )

                    if frame.name == "sleep_level":
                        if frame.has_field("timestamp") and frame.has_field(
                            "sleep_level"
                        ):
                            events.append(
                                {
                                    "time": frame.get_value("timestamp"),
                                    "stage": frame.get_value("sleep_level"),
                                }
                            )

    except Exception as e:
        print(f"Error parsing {os.path.basename(filepath)}: {e}")
        return None

    if not events or not file_end_time:
        return None

    if not stats["date"]:
        stats["date"] = file_end_time.strftime("%Y-%m-%d")

    events.sort(key=lambda x: x["time"])
    detailed_segments = []

    raw_totals = {"deep": 0.0, "light": 0.0, "rem": 0.0, "awake": 0.0, "unknown": 0.0}

    for i in range(len(events)):
        start = events[i]["time"]
        stage = events[i]["stage"]

        if i < len(events) - 1:
            end = events[i + 1]["time"]
        else:
            end = file_end_time

        duration = get_minutes(start, end)

        if duration > 1000 or duration < 0:
            continue

        detailed_segments.append({"stage": stage, "duration": duration})
        if stage in raw_totals:
            raw_totals[stage] += duration
        else:
            raw_totals["unknown"] += duration

    final_stats = raw_totals.copy()

    for i in range(len(detailed_segments) - 1, -1, -1):
        seg = detailed_segments[i]
        if seg["duration"] < 0.5:
            continue

        if seg["stage"] == "awake":
            final_stats["awake"] -= seg["duration"]
            if final_stats["awake"] < 0:
                final_stats["awake"] = 0
            break
        else:
            break

    stats["deep_m"] = round(final_stats["deep"], 1)
    stats["light_m"] = round(final_stats["light"], 1)
    stats["rem_m"] = round(final_stats["rem"], 1)
    stats["awake_m"] = round(final_stats["awake"], 1)

    stats["total_sleep_m"] = stats["deep_m"] + stats["light_m"] + stats["rem_m"]

    stats["time_in_bed_m"] = round(stats["total_sleep_m"] + raw_totals["awake"], 1)

    return stats


def main():
    print("0. Copying Data...")
    copy_files()

    print("1. Indexing HRV Data...")
    hrv_map = parse_hrv_folder(PATH_HRV)
    print(f"   Found {len(hrv_map)} HRV records.")

    print("2. Processing Sleep Files...")
    all_rows = []

    for root, dirs, files in os.walk(PATH_SLEEP):
        for f in sorted(files):
            if f.endswith(".fit"):
                filepath = os.path.join(root, f)
                data = parse_sleep_file(filepath)
                if data:
                    data["hrv_ms"] = hrv_map.get(data["date"], "")
                    all_rows.append(data)

    all_rows.sort(key=lambda x: x["date"])

    headers = [
        "date",
        "sleep_score",
        "recovery_score",
        "hrv_ms",
        "total_sleep_m",
        "deep_m",
        "light_m",
        "rem_m",
        "awake_m",
        "time_in_bed_m",
    ]

    print(f"3. Writing {len(all_rows)} records to {OUTPUT_CSV}...")

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print("Done")


if __name__ == "__main__":
    main()
