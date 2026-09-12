import subprocess, time, json, base64, csv, sys
run_id='34590509770'
repo='bestoddspredictions-web/NFL-PREDICTIONS'
print('Polling run', run_id)
for i in range(120):
    p = subprocess.run(['gh','run','view',run_id,'--repo',repo,'--json','status,conclusion'], capture_output=True, text=True)
    if p.returncode != 0:
        print('gh run view failed:', p.stderr)
        time.sleep(10)
        continue
    try:
        j = json.loads(p.stdout)
    except Exception as e:
        print('json parse error', e, p.stdout)
        time.sleep(10)
        continue
    status = j.get('status')
    conclusion = j.get('conclusion')
    print(i, status, conclusion)
    if status != 'in_progress':
        break
    time.sleep(15)

print('\nFetching run logs...')
subprocess.run(['gh','run','view',run_id,'--repo',repo,'--log'])

print('\nFetching predictions CSV from repo...')
api = f"repos/{repo.replace('/','/')}/contents/data_files/player_props_predictions.csv"
# Correct owner/repo path
owner, repo_name = repo.split('/')
api = f"repos/{owner}/{repo_name}/contents/data_files/player_props_predictions.csv"
p = subprocess.run(['gh','api',api], capture_output=True, text=True)
if p.returncode != 0:
    print('gh api failed:', p.stderr)
    sys.exit(1)
try:
    j = json.loads(p.stdout)
    content_b64 = j.get('content','')
    content = base64.b64decode(content_b64).decode('utf-8', errors='replace')
except Exception as e:
    print('Failed to decode API response:', e)
    sys.exit(1)

# Parse CSV
reader = csv.DictReader(content.splitlines())
rows = list(reader)
if not rows:
    print('Remote predictions file is empty or missing rows')
    sys.exit(0)

dates = []
for r in rows:
    gd = r.get('game_date')
    if gd:
        dates.append(gd)

# print summary
print('\nREMOTE PREDICTIONS SUMMARY:')
print('Total rows:', len(rows))
if dates:
    # naive lexicographic min/max should work with ISO timestamps
    print('Date range:', min(dates), 'to', max(dates))
else:
    print('No game_date values found')

# Print first 20 rows
print('\nFirst 20 predictions:')
for i,r in enumerate(rows[:20]):
    print(i+1, r.get('player_name'), r.get('team'), r.get('prop_type'), r.get('line_value'), r.get('game_date'))

print('\nFull file saved to: ./player_props_predictions_remote.csv')
with open('player_props_predictions_remote.csv','w', encoding='utf-8') as fh:
    fh.write(content)
print('Done')
