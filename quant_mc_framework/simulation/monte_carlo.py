import concurrent.futures
import pandas as pd
import numpy as np
import signac
import itertools
from .engine import LongShortSimulation

np.seterr(all='ignore')

class MonteCarloManager:
    def __init__(self, base_params, param_ranges, n_simulations=100):
        """Initialize Monte Carlo simulations manager.

        Args:
            base_params (dict): Base parameters for all simulations
            param_ranges (dict): Parameters to vary with their ranges
            n_simulations (int): Number of simulations per parameter set
        """
        self.base_params = base_params
        self.param_ranges = param_ranges
        self.n_simulations = n_simulations
        self.project = signac.init_project('quant_mc_simulations')

    
    def setup_jobs(self):
        """Create jobs for all parameter combinations"""
        # Generate all combinations of parameter values
        param_keys = list(self.param_ranges.keys())
        param_values = list(self.param_ranges.values())

        for values in itertools.product(*param_values):
            # Create parameter dictionary for this cobination
            params = self.base_params.copy()
            params.update(dict(zip(param_keys, values)))

            # Create a job for this parameter set
            job = self.project.open_job(params)
            job.init()
            job.doc.setdefault('status', 'initialized')
            job.doc.setdefault('simulations', [])


    def run_job(self, job_id):
        """Run simulations for a specific job"""
        job = self.project.open_job(id=job_id)
        params = job.sp  # Get parameters from job statepoint

        # Run multiple simulations with different seed
        results = []
        for i in range(self.n_simulations):
            seed = hash(f"{job.id}_{i}") % 2**32  # Generate deterministic seed
            simulator = LongShortSimulation(params)
            sim_result = simulator.run(seed=seed)

            # Store only metrics to save space
            results.append(sim_result['metrics'])

        # Store result in job document
        job.doc['simulations'] = results
        job.doc['status'] = 'completed'
        return job.id
    

    def run_all_jobs(self, max_workers=None):
        """Run all jobs in parallel"""
        jobs = list(self.project.find_jobs({'doc.status': 'initialized'}))
        job_ids = [job.id for job in jobs]

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.run_job, job_id) for job_id in job_ids]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    job_id = future.result()
                    print(f"Completed job {job_id}")
                except Exception as e:
                    print(f"Job failed with error: {e}")

    
    def get_results(self, metric_name='sharpe_ratio', strategy='options_overlay'):
        """Get aggregated results for a specific metric"""
        results = {}

        # Get all parameter keys
        param_keys = list(self.param_ranges.keys())

        # For each parameter, collect metrics grouped by parameter value
        for param in param_keys:
            param_values = sorted(self.param_ranges[param])
            metric_by_param = {value: [] for value in param_values}

            for value in param_values:
                # Find jobs with this parameter value
                jobs = self.project.find_jobs({param: value})

                for job in jobs:
                    if job.doc.get('status') == 'completed':
                        # Extract the metric from each simulation
                        for sim in job.doc.get('simulations', []):
                            if strategy in sim and metric_name in sim[strategy]:
                                metric_by_param[value].append(sim[strategy][metric_name])

            results[param] = metric_by_param

        return results