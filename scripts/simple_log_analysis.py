
import re
import sys

def analyze_latest_run(log_path):
    print(f"Analyzing {log_path}...")
    
    tasks = {}
    assessment_id = None
    
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
            
        # 1. Find the last "Assessment completed" line
        last_summary_idx = -1
        for i in range(len(lines) - 1, -1, -1):
            if "Assessment completed" in lines[i]:
                last_summary_idx = i
                break
        
        if last_summary_idx == -1:
            print("No completed assessment found in logs.")
            return

        print(f"Found latest assessment at line {last_summary_idx}")

        # 2. Search backwards from summary for task completions
        for i in range(last_summary_idx, -1, -1):
            line = lines[i]
            
            if "Starting assessment" in line:
                break
                
            if "Task completed" in line:
                # Naive line scan for key=value
                tid = ""
                score = 0.0
                success = False
                
                # Check current and next few lines for multi-line log
                chunk = "".join(lines[i:i+15])
                
                m_tid = re.search(r'task_id=([a-z0-9_]+)', chunk)
                if m_tid: tid = m_tid.group(1)
                
                m_score = re.search(r'score=([0-9\.]+)', chunk)
                if m_score: score = float(m_score.group(1))
                
                m_succ = re.search(r'success=(True|False)', chunk)
                if m_succ: success = (m_succ.group(1) == "True")
                
                if tid and tid not in tasks:
                    tasks[tid] = {"score": score, "success": success}

        # 3. Print Report
        print("\n=== LATEST RUN RESULTS ===")
        sorted_ids = sorted(tasks.keys())
        success_count = 0
        
        for tid in sorted_ids:
            t = tasks[tid]
            icon = "✅" if t["success"] else "❌"
            if t["success"]: success_count += 1
            print(f"{icon} {tid}: Score={t['score']:.2f}")
            
        print(f"\nTotal: {len(tasks)}")
        print(f"Successful: {success_count}")
        print(f"Success Rate: {success_count/len(tasks)*100:.1f}%" if tasks else "0%")

    except FileNotFoundError:
        print("Log file not found.")

if __name__ == "__main__":
    path = "/Users/nikhilpujari/agentbeats/agentbeats-leaderboard-template/green.log"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    analyze_latest_run(path)
