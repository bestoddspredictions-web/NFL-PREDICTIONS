import subprocess, time
run_id='34590509770'
repo='bestoddspredictions-web/NFL-PREDICTIONS'
for i in range(60):
    p = subprocess.run(["gh","run","view",run_id,"--repo",repo,"--json","status,conclusion","--jq",".status + ' ' + (.conclusion // \"null\")"], capture_output=True, text=True)
    s = p.stdout.strip()
    print(i, s)
    if 'completed' in s:
        break
    time.sleep(20)
# fetch logs
subprocess.run(["gh","run","view",run_id,"--repo",repo,"--log"]) 
