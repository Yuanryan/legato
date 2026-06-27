import json
import concurrent.futures
import xml.etree.ElementTree as ET
import os
from tqdm import tqdm
from datasets import load_from_disk
from argparse import ArgumentParser
from utils.TEDn_eval.evaluation.TEDn_xml_xml import TEDn_xml_xml

def cut_xml(xml, limit=6000):
    root = ET.fromstring(xml)
    num_nodes = sum(1 for _ in root.iter())
    initial_num_nodes = num_nodes

    while num_nodes > limit:
        parts = root.findall('part')
        for part in parts:
            part.remove(part[-1])
        num_nodes = sum(1 for _ in root.iter())

    return ET.tostring(root, encoding='unicode'), initial_num_nodes
        
def compute_score(input_data):
    idx, pred_xml, gold_xml = input_data
    if not pred_xml or not pred_xml.strip():
        return idx, 100.0
    try:
        pred_xml, pred_init_num_nodes = cut_xml(pred_xml, 6000)
    except ET.ParseError:
        return idx, 100.0
    gold_xml, gold_init_num_nodes = cut_xml(gold_xml, 6000)
    if pred_init_num_nodes > 6000:
        print(f"Index {idx}: Pred XML initial nodes {pred_init_num_nodes}, after cut {sum(1 for _ in ET.fromstring(pred_xml).iter())}")
    if gold_init_num_nodes > 6000:
        print(f"Index {idx}: Gold XML initial nodes {gold_init_num_nodes}, after cut {sum(1 for _ in ET.fromstring(gold_xml).iter())}")
    score = TEDn_xml_xml(pred_xml, gold_xml, flavor='lmx')
    return idx, score.edit_cost / score.gold_cost * 100

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--prediction_file", type=str, help="Path to the XML prediction JSON file")
    parser.add_argument("--ground_truth", type=str, help="Path to the dataset or ground truth file")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for parallel processing")

    args = parser.parse_args()
    
    if os.path.basename(args.prediction_file).split('.')[0].split('_')[-1] != 'xml':
        print("[WARNING] The prediction file name is not ended with `xml`. TEDn computation may fail.")

    with open(args.prediction_file, "r") as f:
        pred_xmls = json.load(f)

    if args.ground_truth.endswith('.json'):
        with open(args.ground_truth, "r") as f:
            gold_xmls = json.load(f)
    else:
        ds = load_from_disk(args.ground_truth)
        gold_xmls = [txt for txt in ds['musicxml']]

    # Process items in batches so a single OOM-killed worker only takes down its batch,
    # not the entire run. We also catch BrokenProcessPool to allow resuming with a fresh pool.
    tuples = list(zip(range(len(pred_xmls)), pred_xmls, gold_xmls))
    TED_scores = []
    failed_indices = []
    batch_size = max(args.num_workers * 4, 16)
    pending = list(tuples)
    pbar = tqdm(total=len(tuples), desc="Computing TED scores...")
    while pending:
        batch, pending = pending[:batch_size], pending[batch_size:]
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
                future_to_idx = {executor.submit(compute_score, data): data[0] for data in batch}
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        TED_scores.append(future.result())
                    except Exception as e:
                        print(f"Index {idx}: failed with {type(e).__name__}: {e}; assigning 100% score")
                        failed_indices.append(idx)
                        TED_scores.append((idx, 100.0))
                    pbar.update(1)
        except concurrent.futures.process.BrokenProcessPool:
            # Mark all unfinished items in this batch as failed
            done_indices = {idx for idx, _ in TED_scores}
            for idx, _, _ in batch:
                if idx not in done_indices:
                    print(f"Index {idx}: worker pool broken; assigning 100% score")
                    failed_indices.append(idx)
                    TED_scores.append((idx, 100.0))
                    pbar.update(1)
    pbar.close()
    if failed_indices:
        print(f"Failed indices ({len(failed_indices)}): {sorted(failed_indices)}")
    TED_scores.sort(key=lambda x: x[0])
    TED_scores = [x[1] for x in TED_scores]

    scores = {"average_TEDn":  sum(TED_scores) / len(TED_scores), "all_TEDn": TED_scores}
    print(f"Average TED score: {scores['average_TEDn']}")

    output_file = args.prediction_file.replace(".json", "_ted_scores.json")
    print(f"Saving TED scores to {output_file}")
    with open(output_file, "w") as f:
        json.dump(scores, f)
    print("Done.")
