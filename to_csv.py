import os
import csv
import fitdecode
from datetime import datetime, timedelta

# ==========================================
# CONFIGURATION
# ==========================================
# Paths to your Gadgetbridge export directories
PATH_SLEEP = './Garmin/SLEEP'
PATH_HRV   = './Garmin/HRV_STATUS'
OUTPUT_CSV = 'daily_health_stats.csv'

def get_minutes(start, end):
    return (end - start).total_seconds() / 60

def parse_hrv_folder(folder_path):
    """Scans HRV folder to map Date -> Overnight HRV (ms)."""
    hrv_map = {}
    if not os.path.exists(folder_path):
        return hrv_map

    for root, dirs, files in os.walk(folder_path):
        for f in files:
            if not f.endswith('.fit'): continue
            
            try:
                with fitdecode.FitReader(os.path.join(root, f)) as fit:
                    for frame in fit:
                        if isinstance(frame, fitdecode.FitDataMessage) and frame.name == 'hrv_status_summary':
                            if frame.has_field('weekly_average'): 
                                # Note: 'weekly_average' or 'last_night_average' depending on device
                                # Usually 'last_night_average' is what we want for daily stats
                                val = frame.get_value('last_night_average')
                                if val is None: val = frame.get_value('weekly_average')
                                
                                # Timestamp for the key
                                if frame.has_field('timestamp'):
                                    ts = frame.get_value('timestamp')
                                    # Key by YYYY-MM-DD
                                    date_key = ts.strftime('%Y-%m-%d')
                                    hrv_map[date_key] = val
            except:
                continue
    return hrv_map

def parse_sleep_file(filepath):
    """Extracts stages with Tail-Trim logic and Assessment scores."""
    events = []
    file_end_time = None
    
    # Metrics to extract
    stats = {
        'date': None,
        'sleep_score': None,
        'recovery_score': None, # Proxy for Body Battery Gain
        'deep_m': 0.0,
        'light_m': 0.0,
        'rem_m': 0.0,
        'awake_m': 0.0,
        'total_sleep_m': 0.0, # Excluding Awake
        'time_in_bed_m': 0.0  # Including Awake
    }

    try:
        with fitdecode.FitReader(filepath) as fit:
            for frame in fit:
                if isinstance(frame, fitdecode.FitDataMessage):
                    
                    # 1. Global Timestamp (End Anchor)
                    if frame.has_field('timestamp'):
                        file_end_time = frame.get_value('timestamp')

                    # 2. Assessment Scores
                    if frame.name == 'sleep_assessment':
                        if frame.has_field('overall_sleep_score'):
                            stats['sleep_score'] = frame.get_value('overall_sleep_score')
                        if frame.has_field('sleep_recovery_score'):
                            stats['recovery_score'] = frame.get_value('sleep_recovery_score')
                        # Use assessment timestamp as the "Date" of the sleep record
                        if frame.has_field('timestamp'):
                            stats['date'] = frame.get_value('timestamp').strftime('%Y-%m-%d')

                    # 3. Sleep Events
                    if frame.name == 'sleep_level':
                        if frame.has_field('timestamp') and frame.has_field('sleep_level'):
                            events.append({
                                'time': frame.get_value('timestamp'),
                                'stage': frame.get_value('sleep_level')
                            })
                            
    except Exception as e:
        print(f"Error parsing {os.path.basename(filepath)}: {e}")
        return None

    if not events or not file_end_time:
        return None

    # If date wasn't found in assessment, use the file end time
    if not stats['date']:
        stats['date'] = file_end_time.strftime('%Y-%m-%d')

    # --- CALCULATE DURATIONS ---
    events.sort(key=lambda x: x['time'])
    detailed_segments = []
    
    raw_totals = {'deep': 0.0, 'light': 0.0, 'rem': 0.0, 'awake': 0.0, 'unknown': 0.0}

    for i in range(len(events)):
        start = events[i]['time']
        stage = events[i]['stage']
        
        if i < len(events) - 1:
            end = events[i+1]['time']
        else:
            end = file_end_time

        duration = get_minutes(start, end)
        
        # Guard against massive gaps or errors
        if duration > 1000 or duration < 0: continue

        detailed_segments.append({'stage': stage, 'duration': duration})
        if stage in raw_totals: raw_totals[stage] += duration
        else: raw_totals['unknown'] += duration

    # --- APPLY TAIL TRIM LOGIC ---
    # Clone for "App Logic" calculation
    final_stats = raw_totals.copy()
    
    # Iterate backwards to find the last meaningful segment
    for i in range(len(detailed_segments) - 1, -1, -1):
        seg = detailed_segments[i]
        if seg['duration'] < 0.5: continue # Skip markers
        
        # If the last real block is Awake, trim it from stats (but keep in Time in Bed)
        if seg['stage'] == 'awake':
            final_stats['awake'] -= seg['duration']
            # Ensure we don't go negative due to float math
            if final_stats['awake'] < 0: final_stats['awake'] = 0
            break # Only trim the very last block
        else:
            break

    # Populate Result Dictionary
    stats['deep_m']   = round(final_stats['deep'], 1)
    stats['light_m']  = round(final_stats['light'], 1)
    stats['rem_m']    = round(final_stats['rem'], 1)
    stats['awake_m']  = round(final_stats['awake'], 1)
    
    # Total Sleep = Deep + Light + REM
    stats['total_sleep_m'] = stats['deep_m'] + stats['light_m'] + stats['rem_m']
    
    # Time in Bed = Total Sleep + All Awake Time (Including the trimmed tail)
    # Note: raw_totals['awake'] includes the tail
    stats['time_in_bed_m'] = round(stats['total_sleep_m'] + raw_totals['awake'], 1)

    return stats

def main():
    print("1. Indexing HRV Data...")
    hrv_map = parse_hrv_folder(PATH_HRV)
    print(f"   Found {len(hrv_map)} HRV records.")

    print("2. Processing Sleep Files...")
    all_rows = []
    
    for root, dirs, files in os.walk(PATH_SLEEP):
        for f in sorted(files):
            if f.endswith('.fit'):
                filepath = os.path.join(root, f)
                data = parse_sleep_file(filepath)
                if data:
                    # Merge HRV if available
                    data['hrv_ms'] = hrv_map.get(data['date'], '')
                    all_rows.append(data)

    # 3. Sort by Date
    all_rows.sort(key=lambda x: x['date'])

    # 4. Write CSV
    headers = ['date', 'sleep_score', 'recovery_score', 'hrv_ms', 
               'total_sleep_m', 'deep_m', 'light_m', 'rem_m', 'awake_m', 'time_in_bed_m']
    
    print(f"3. Writing {len(all_rows)} records to {OUTPUT_CSV}...")
    
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print("✅ Done.")

if __name__ == "__main__":
    main()
