import json
import re
from collections import defaultdict

def run_analysis():
    file_path = 'persona_test_results.json'
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Run your simulation script first.")
        return

    # metrics[persona][policy] = {counts}
    metrics = defaultdict(lambda: defaultdict(lambda: {
        'total': 0, 
        'unnecessary_count': 0, 
        'stop_count': 0, 
        'concern_total': 0,
        'failed_comprehension': 0
    }))

    for entry in data:
        if 'error' in entry: continue
        
        p = entry['persona']
        pol = entry['policy']
        resp = entry['response'].lower()
        
        m = metrics[p][pol]
        m['total'] += 1
        
        # 1. Necessity check (Matches Q2)
        if re.search(r'2\..*no', resp):
            m['unnecessary_count'] += 1
            
        # 2. Action/Stop check (Matches Q3)
        if re.search(r'3\..*yes', resp):
            m['stop_count'] += 1
            
        # 3. Concern Likert (Matches Q4)
        concern_match = re.search(r'4\..*(\d)', resp)
        if concern_match:
            m['concern_total'] += int(concern_match.group(1))
            
        # 4. Simple Comprehension Check (Matches Q1)
        # Check if they failed to identify specific data types (simulating the 55% gap)
        # e.g., if Normal User just says "collect info" without naming specifics
        summary = resp.split('\n')[0]
        if len(summary.split()) < 3 or "collect" in summary and "data" in summary and len(summary.split()) < 5:
            m['failed_comprehension'] += 1

    # Header for Output Table
    print(f"{'Persona':<22} | {'Policy':<12} | {'Unnecessary %':<15} | {'Stop %':<10} | {'Avg Concern'}")
    print("-" * 85)

    for persona, policies in metrics.items():
        for policy, m in policies.items():
            total = m['total']
            un_pct = (m['unnecessary_count'] / total) * 100
            stop_pct = (m['stop_count'] / total) * 100
            avg_concern = m['concern_total'] / total
            
            print(f"{persona:<22} | {policy:<12} | {un_pct:>13.1f}% | {stop_pct:>8.1f}% | {avg_concern:>11.2f}")

if __name__ == "__main__":
    run_analysis()