import os
import re
import csv
from pathlib import Path

def extract_instance_name_from_filename(filename):
    """Extract testcase name from filename by removing extension"""
    return Path(filename).stem

def read_input_file(filepath):
    """Read input .sol file and extract number of vehicles and optimal distance"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Count number of routes (vehicles used)
    routes = re.findall(r'Route #\d+:', content)
    num_vehicles = len(routes)
    
    # Extract cost/distance
    cost_match = re.search(r'Cost\s+(\d+)', content)
    optimal_distance = int(cost_match.group(1)) if cost_match else None
    
    # Extract instance size (number of nodes/customers)
    # Count all node numbers in routes (excluding the depot if it's 1)
    node_pattern = r'Route #\d+:\s*(.*?)(?:\n|$)'
    all_nodes = []
    for match in re.finditer(node_pattern, content):
        route_nodes = match.group(1).strip().split()
        all_nodes.extend(route_nodes)
    
    # Assuming node numbers are customers (excluding depot if present)
    instance_size = len(set(all_nodes))
    
    return {
        'num_vehicles': num_vehicles,
        'optimal_distance': optimal_distance,
        'instance_size': instance_size
    }

def read_bc_file(filepath):
    """Read branch-and-cut output file and extract distance"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Extract total distance
    distance_match = re.search(r'Total distance:\s*(\d+)', content)
    if not distance_match:
        # Try alternative format
        distance_match = re.search(r'Distance:\s*(\d+)', content)
    
    bc_distance = int(distance_match.group(1)) if distance_match else None
    return bc_distance

def read_ortools_file(filepath):
    """Read OR-Tools output file and extract distance"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Extract total distance
    distance_match = re.search(r'Total distance:\s*(\d+)', content)
    ortools_distance = int(distance_match.group(1)) if distance_match else None
    
    return ortools_distance

def calculate_gap(optimal, solution):
    """Calculate gap percentage between optimal and solution"""
    if optimal is None or solution is None or optimal == 0:
        return None
    return round(((solution - optimal) / optimal) * 100, 2)

def main():
    # Define directories - UPDATE THESE PATHS
    input_dir = "../../Dataset/A"
    out1_dir = "../outputs/b&c"
    out2_dir = "../outputs/Ortools"
    
    # Dictionary to store all results
    results = []
    
    # Get all .sol files from input_dir
    if not os.path.exists(input_dir):
        print(f"Error: Directory '{input_dir}' not found!")
        return
    
    sol_files = [f for f in os.listdir(input_dir) if f.endswith('.sol')]
    
    if not sol_files:
        print(f"No .sol files found in '{input_dir}'")
        return
    
    print(f"Found {len(sol_files)} .sol files")
    print(f"Looking for B&C files in: {out1_dir}")
    print(f"Looking for OR-Tools files in: {out2_dir}")
    print("\n" + "="*80)
    
    for sol_file in sorted(sol_files):  # Sort input files
        testcase_name = extract_instance_name_from_filename(sol_file)
        print(f"\nProcessing: {testcase_name}")
        
        # Read input file
        input_filepath = os.path.join(input_dir, sol_file)
        input_data = read_input_file(input_filepath)
        print(f"  - Instance size: {input_data['instance_size']}")
        print(f"  - Vehicles: {input_data['num_vehicles']}")
        print(f"  - Optimal distance: {input_data['optimal_distance']}")
        
        # Initialize with default values
        bc_distance = None
        ortools_distance = None
        
        # Look for corresponding b&c output file (with _solution.txt suffix)
        bc_filename = f"{testcase_name}_solution.txt"
        bc_filepath = os.path.join(out1_dir, bc_filename)
        
        if os.path.exists(bc_filepath):
            bc_distance = read_bc_file(bc_filepath)
            print(f"  ✓ Found B&C file: {bc_filename}")
            print(f"    - Distance: {bc_distance}")
        else:
            print(f"  ✗ B&C file not found: {bc_filename}")
        
        # Look for corresponding OR-Tools output file (with _solution.txt suffix)
        ortools_filename = f"{testcase_name}_solution.txt"
        ortools_filepath = os.path.join(out2_dir, ortools_filename)
        
        if os.path.exists(ortools_filepath):
            ortools_distance = read_ortools_file(ortools_filepath)
            print(f"  ✓ Found OR-Tools file: {ortools_filename}")
            print(f"    - Distance: {ortools_distance}")
        else:
            print(f"  ✗ OR-Tools file not found: {ortools_filename}")
        
        # Calculate gaps
        bc_gap = calculate_gap(input_data['optimal_distance'], bc_distance)
        ortools_gap = calculate_gap(input_data['optimal_distance'], ortools_distance)
        
        # Store results
        results.append({
            'testcase_name': testcase_name,
            'instance_size': input_data['instance_size'],
            'num_of_vehicles': input_data['num_vehicles'],
            'optimal_distance': input_data['optimal_distance'],
            'b&c_distance': bc_distance if bc_distance is not None else '',
            'b&c_gap': bc_gap if bc_gap is not None else '',
            'ortools_distance': ortools_distance if ortools_distance is not None else '',
            'ortools_gap': ortools_gap if ortools_gap is not None else ''
        })
    
    # Sort results by testcase_name
    results.sort(key=lambda x: x['testcase_name'])
    
    # Write results to CSV
    output_file = '../outputs/results.csv'
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['testcase_name', 'instance_size', 'num_of_vehicles', 
                     'optimal_distance', 'b&c_distance', 'b&c_gap', 
                     'ortools_distance', 'ortools_gap']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"\n{'='*80}")
    print(f"✓ Results written to {output_file}")
    print(f"✓ Processed {len(results)} test cases")
    
    # Summary statistics
    bc_found = sum(1 for r in results if r['b&c_distance'] != '')
    ortools_found = sum(1 for r in results if r['ortools_distance'] != '')
    bc_with_gap = sum(1 for r in results if r['b&c_gap'] != '')
    ortools_with_gap = sum(1 for r in results if r['ortools_gap'] != '')
    
    print(f"\nSummary:")
    print(f"  - B&C files found: {bc_found}/{len(results)}")
    print(f"  - B&C gaps calculated: {bc_with_gap}/{len(results)}")
    print(f"  - OR-Tools files found: {ortools_found}/{len(results)}")
    print(f"  - OR-Tools gaps calculated: {ortools_with_gap}/{len(results)}")
    
    # Show first few rows of CSV
    print(f"\nPreview of results (first 5 rows):")
    print(f"{'-'*80}")
    with open(output_file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines[:6]):  # Header + first 5 rows
            print(line.strip())

if __name__ == "__main__":
    main()