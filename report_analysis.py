import argparse
import csv
import json
import platform
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

METHODS = [
    'Greedy (N=1)',
    'Self-Consistency (N=5)',
    'Compute-Matched SC (N=15)',
    'PRM Best-of-5 (min)',
    'PRM Best-of-5 (product)',
    'PRM Best-of-5 (step_penalty, lambda=2.0)',
]


def load_json(path):
    with path.open(encoding='utf-8') as f:
        return json.load(f)


def load_records(path):
    with path.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_csv(path, rows, fields=None):
    if not rows:
        return
    fields = fields or list(rows[0])
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def correctness(records, method):
    return np.array([int(r['selected_answers'][method] == r['ground_truth_answer']) for r in records], dtype=int)


def bootstrap_ci(values, rng, resamples=1000):
    n = len(values)
    draws = rng.integers(0, n, size=(resamples, n))
    means = values[draws].mean(axis=1) * 100
    return np.percentile(means, [2.5, 97.5]).tolist()


def prepare_audit(records, dest):
    existing = {}
    if dest.exists():
        with dest.open(newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                existing[(row['problem_id'], row['candidate_index'], row['step_index'])] = row

    pool = []
    for record in records:
        for ci, candidate in enumerate(record['candidates'][:5]):
            steps = candidate.get('steps', [])
            scores = candidate.get('step_scores', [])
            rationales = candidate.get('rationales', [])
            for si, (step, score) in enumerate(zip(steps, scores)):
                if isinstance(score, str):
                    label = score.strip().upper()
                    if label in ('VALID', '1', '+1', 'TRUE'):
                        prediction = 'VALID'
                    elif label in ('INVALID', '0', '-1', 'FALSE'):
                        prediction = 'INVALID'
                    else:
                        continue
                else:
                    prediction = 'VALID' if float(score) > 0 else 'INVALID'
                pool.append({
                    'problem_id': str(record['id']),
                    'candidate_index': str(ci),
                    'step_index': str(si),
                    'question': record['question'],
                    'reasoning_prefix': '\n'.join(str(s) for s in steps[:si]),
                    'step': str(step),
                    'judge_prediction': prediction,
                    'judge_rationale': str(rationales[si]) if si < len(rationales) else '',
                    'manual_label': '',
                    'notes': '',
                })

    invalid = [r for r in pool if r['judge_prediction'] == 'INVALID']
    valid = [r for r in pool if r['judge_prediction'] == 'VALID']
    selected = invalid[:min(15, len(invalid))] + valid[:max(0, 30 - min(15, len(invalid)))]
    if len(selected) < 30:
        selected += invalid[min(15, len(invalid)):min(15, len(invalid)) + (30 - len(selected))]
    selected = selected[:30]
    for row in selected:
        key = (row['problem_id'], row['candidate_index'], row['step_index'])
        if key in existing:
            row['manual_label'] = existing[key].get('manual_label', '')
            row['notes'] = existing[key].get('notes', '')
    write_csv(dest, selected)
    return len(invalid), len(selected), sum(r['judge_prediction'] == 'INVALID' for r in selected)


def audit_metrics(path):
    if not path.exists():
        return None
    with path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    labeled = [r for r in rows if r['manual_label'].strip().upper() in ('VALID', 'INVALID')]
    if not labeled:
        return None
    tp = sum(r['judge_prediction'] == 'INVALID' and r['manual_label'].strip().upper() == 'INVALID' for r in labeled)
    fp = sum(r['judge_prediction'] == 'INVALID' and r['manual_label'].strip().upper() == 'VALID' for r in labeled)
    tn = sum(r['judge_prediction'] == 'VALID' and r['manual_label'].strip().upper() == 'VALID' for r in labeled)
    fn = sum(r['judge_prediction'] == 'VALID' and r['manual_label'].strip().upper() == 'INVALID' for r in labeled)
    return {'labeled': len(labeled), 'judge_invalid_labeled': tp + fp,
            'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
            'precision_invalid': tp / (tp + fp) if tp + fp else None,
            'recall_invalid': tp / (tp + fn) if tp + fn else None}


def plot(x, y, prm_x, prm_y, xlabel, destination):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(x, y, marker='o', label='Self-Consistency N=1..15')
    for name, xx, yy in zip(METHODS[3:], prm_x, prm_y):
        ax.scatter(xx, yy, s=65, label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Accuracy (%)')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs', type=Path, default=Path('outputs'))
    parser.add_argument('--report-dir', type=Path, default=Path('report_assets'))
    args = parser.parse_args()
    source, target = args.outputs, args.report_dir
    target.mkdir(parents=True, exist_ok=True)
    records = load_records(source / 'per_problem.jsonl')
    summary = load_json(source / 'results_summary.json')
    ablation = load_json(source / 'aggregation_ablation.json')
    scaling = load_json(source / 'sc_scaling_curve.json')
    if len(records) != 50 or len({r['id'] for r in records}) != 50:
        raise ValueError('Expected exactly 50 unique problem records. Finish the benchmark first.')
    if len(summary) != 6 or len(ablation) != 5 or len(scaling) != 15:
        raise ValueError('Unexpected summary, ablation or scaling row count.')
    if not all(len(r['candidates']) == 15 for r in records):
        raise ValueError('Each problem must contain exactly 15 sampled candidates.')
    by_method = {r['Method']: r for r in summary}
    if set(METHODS) != set(by_method):
        raise ValueError('Method names do not match the expected six configurations.')
    for n, method in [(5, METHODS[1]), (15, METHODS[2])]:
        actual = float(scaling[n - 1]['Accuracy (%)'])
        expected = float(by_method[method]['Accuracy (%)'])
        if abs(actual - expected) > 0.011:
            raise ValueError(f'SC@{n} does not match scaling curve.')
    keys = ['Generation Calls', 'Judge Calls', 'Total LLM Calls',
            'Avg Gen Tokens / Problem', 'Avg Judge Tokens / Problem']
    for key in keys:
        if len({by_method[m][key] for m in METHODS[3:]}) != 1:
            raise ValueError(f'PRM compute reuse inconsistent: {key}')

    rng = np.random.default_rng(5326)
    accuracy_rows = []
    for method in METHODS:
        vals = correctness(records, method)
        lo, hi = bootstrap_ci(vals, rng)
        accuracy_rows.append({'Method': method, 'Accuracy (%)': round(100 * vals.mean(), 2),
                              'CI Low (%)': round(lo, 2), 'CI High (%)': round(hi, 2)})
    write_csv(target / 'bootstrap_accuracy.csv', accuracy_rows)

    pairs = []
    for prm in METHODS[3:]:
        p = correctness(records, prm)
        for baseline in (METHODS[1], METHODS[2]):
            b = correctness(records, baseline)
            diff = (p - b).astype(float)
            lo, hi = bootstrap_ci(diff, rng)
            pairs.append({'PRM Method': prm, 'Baseline': baseline,
                          'PRM Wins': int(((p == 1) & (b == 0)).sum()),
                          'PRM Losses': int(((p == 0) & (b == 1)).sum()),
                          'Both Correct': int(((p == 1) & (b == 1)).sum()),
                          'Both Wrong': int(((p == 0) & (b == 0)).sum()),
                          'Accuracy Difference (pp)': round(100 * diff.mean(), 2),
                          'Difference CI Low (pp)': round(lo, 2),
                          'Difference CI High (pp)': round(hi, 2)})
    write_csv(target / 'paired_comparisons.csv', pairs)

    n_problems = len(records)
    sc_call_x = [n * n_problems for n in range(1, 16)]
    sc_accuracy = [float(r['Accuracy (%)']) for r in scaling]
    prm_calls = [float(by_method[m]['Total LLM Calls']) for m in METHODS[3:]]
    prm_accuracy = [float(by_method[m]['Accuracy (%)']) for m in METHODS[3:]]
    plot(sc_call_x, sc_accuracy, prm_calls, prm_accuracy,
         'Total LLM calls over 50 problems', target / 'accuracy_vs_calls.png')

    sc_tokens = [float(r['Avg Gen Tokens / Problem']) * n_problems for r in scaling]
    prm_tokens = [(float(by_method[m]['Avg Gen Tokens / Problem']) +
                   float(by_method[m]['Avg Judge Tokens / Problem'])) * n_problems for m in METHODS[3:]]
    plot(sc_tokens, sc_accuracy, prm_tokens, prm_accuracy,
         'Total output tokens over 50 problems (generator + judge)',
         target / 'accuracy_vs_tokens.png')

    greedy_total = sum(int(r['greedy']['generated_tokens']) for r in records)
    budget = 10 * greedy_total
    budget_rows = []
    for n, tokens, acc in zip(range(1, 16), sc_tokens, sc_accuracy):
        budget_rows.append({'Method': f'SC@{n}', 'Output Tokens': round(tokens),
                            'Accuracy (%)': acc, 'Within Budget': tokens <= budget})
    for method, tokens in zip(METHODS[3:], prm_tokens):
        budget_rows.append({'Method': method, 'Output Tokens': round(tokens),
                            'Accuracy (%)': float(by_method[method]['Accuracy (%)']),
                            'Within Budget': tokens <= budget})
    write_csv(target / 'compute_budget.csv', budget_rows)

    audit_path = target / 'manual_audit.csv'
    invalid_available, audit_count, audit_invalid = prepare_audit(records, audit_path)
    audit = audit_metrics(audit_path)
    environment = {'python': sys.version.split()[0], 'platform': platform.platform(),
                   'analysis_numpy': np.__version__, 'analysis_matplotlib': matplotlib.__version__}
    (target / 'analysis_environment.json').write_text(json.dumps(environment, indent=2) + '\n')

    report = [
        '# Report analysis outputs',
        f'Problems evaluated: {n_problems}',
        f'Exact greedy generated tokens: {greedy_total}',
        f'10x greedy token budget: {budget}',
        f'Available judge-INVALID steps: {invalid_available}',
        f'Prepared audit steps: {audit_count} (judge-INVALID: {audit_invalid})',
        '', '## Bootstrap accuracy intervals (1000 resamples, seed 5326)',
    ]
    for row in accuracy_rows:
        report.append(f"- {row['Method']}: {row['Accuracy (%)']}% [{row['CI Low (%)']}%, {row['CI High (%)']}%]")
    report.extend(['', '## Paired comparisons'])
    for row in pairs:
        report.append(f"- {row['PRM Method']} vs {row['Baseline']}: wins {row['PRM Wins']}, losses {row['PRM Losses']}, difference {row['Accuracy Difference (pp)']} pp, 95% CI [{row['Difference CI Low (pp)']}, {row['Difference CI High (pp)']}]")
    report.extend(['', '## Manual audit'])
    if audit:
        report.append(json.dumps(audit, indent=2))
    else:
        report.append('Fill manual_label with VALID or INVALID in manual_audit.csv, then rerun.')
    if invalid_available < 10:
        report.append('WARNING: Fewer than 10 judge-INVALID steps available; audit requirement cannot be met from this dataset.')
    if audit_invalid < 10:
        report.append('WARNING: Audit sample contains fewer than 10 judge-INVALID steps.')
    report.extend(['', '## Remaining manual report tasks',
                   '- Review 3 real failure cases: judge false positive, false negative, and mode collapse/shared flawed premise.',
                   '- Interpret aggregation_ablation.json and its measured tie rates.',
                   '- Compare methods within the 10x greedy output-token budget using compute_budget.csv.',
                   '- Record actual Kaggle GPU runtime, model versions, package versions, and limitations.',
                   '- Write the five report sections and export REPORT.pdf (6 pages maximum).'])
    (target / 'report_notes.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print('Saved report analysis to:', target)
    for path in sorted(target.iterdir()):
        print(' ', path.name)
    print(f'10x greedy output-token budget: {budget}')
    print(f'Manual audit: {audit_count} steps, {audit_invalid} judge-INVALID. Enter manual labels before reporting precision/recall.')


if __name__ == '__main__':
    main()
