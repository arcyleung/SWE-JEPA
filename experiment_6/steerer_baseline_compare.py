import json

def get_f2p_p2p(cases):
    # print(cases[0].keys())
    p2p = sum([1 if c.get("success_p2p") else 0 for c in cases])
    f2p = sum([1 if c.get("success_f2p") else 0 for c in cases])

    return p2p, f2p

featbench_outputs = {
"baseline": "/shared_workspace_mfs/arthur/coder/eval/FeatBench/docker_agent/evaluation_results_baseline_phase6_2.json",
"steered_v1": "/shared_workspace_mfs/arthur/coder/eval/FeatBench/docker_agent/evaluation_results_steered_v1.json",
"steered_v2": "/shared_workspace_mfs/arthur/coder/eval/FeatBench/docker_agent/evaluation_results_steered_phase6_2.json",
"steered_v3": "/shared_workspace_mfs/arthur/coder/eval/FeatBench/docker_agent/evaluation_results_steered_v3.json"
}

for k, v in featbench_outputs.items():
    cases = json.load(open(v))
    print(f"""{k}: {get_f2p_p2p(cases)} {len(cases)}""")