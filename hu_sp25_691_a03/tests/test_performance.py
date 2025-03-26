import time
import logging
import subprocess

def measure_time(command):
    """Runs a subprocess command and measures execution time."""
    start_time = time.time()
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    end_time = time.time()
    execution_time = end_time - start_time
    return execution_time, result.returncode, result.stdout, result.stderr

def benchmark_pipeline():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    steps = [
        "step01_ingest --input_filename 'all'",
        "step02_generate_embeddings --input_filename 'all'",
        "step03_store_vectors --input_filename 'all'",
        "step04_retrieve_chunks --query_args 'Tell me about US constitution'",
        "step05_generate_response --query_args 'Tell me about US constitution' --use_rag"
    ]
    for step in steps:
        logging.info(f"Running {step}...")
        exec_time, returncode, stdout, stderr = measure_time(f"python3 main.py {step}")
        if returncode != 0:
            logging.error(f"Error in {step}: {stderr}")
        else:
            if stderr:
                logging.warning(f"Warnings in {step}: {stderr}")
            logging.info(f"{step} completed in {exec_time:.2f} seconds.")
            logging.debug(stdout)

if __name__ == "__main__":
    benchmark_pipeline()