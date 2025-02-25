import json
import argparse
from pointer_to_penman import convert_amr

def convert_file(input_file: str, output_file: str):
    """Convert all AMRs in a JSONL file from pointer to Penman notation."""
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            data = json.loads(line)
            if 'amr' in data:
                # Convert AMR to Penman notation
                data['amr'] = convert_amr(data['amr'])
            # Write the modified JSON line
            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

def main():
    parser = argparse.ArgumentParser(description='Convert pointer-based AMR to Penman notation')
    parser.add_argument('input_file', help='Input JSONL file containing AMRs')
    parser.add_argument('output_file', help='Output JSONL file to write converted AMRs')
    
    args = parser.parse_args()
    convert_file(args.input_file, args.output_file)

if __name__ == '__main__':
    main()
