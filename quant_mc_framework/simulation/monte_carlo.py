import concurrent.futures
import pandas as pd
import numpy as np
import signac
import itertools
from typing import Dict, List, Any, Optional
from .engine import LongShortSimulation


class MonteCarloManager:
    def __init__(self, base_params: Dict[str, Any], param_ranges: Dict[str, List[Any]], n_simulations: int = 100):
        """
        Initialize Monte Carlo simulations manager.
        
        Parameters
        ----------
        base_params : Dict[str, Any]
            Base parameters for all simulations
        param_ranges : Dict[str, List[Any]]
            Parameters to vary with their ranges
        n_simulations : int, default=100
            Number of simulations per parameter set
        """
        self.base_params = base_params
        self.param_ranges = param_ranges
        self.n_simulations = n_simulations
        self.project = signac.init_project('quant_mc_simulations')

    
    def setup_jobs(self) -> None:
        """
        Create jobs for all parameter combinations.
        
        This method generates all possible combinations of the parameters
        specified in param_ranges and creates a job for each combination.
        """
        # Generate all combinations of parameter values
        param_keys = list(self.param_ranges.keys())
        param_values = list(self.param_ranges.values())

        # Track number of jobs created
        jobs_created = 0

        for values in itertools.product(*param_values):
            # Create parameter dictionary for this cobination
            params = self.base_params.copy()

            # Update the parameters that are being varied
            for i, key in enumerate(param_keys):
                params[key] = values[i]

            # Create a job for this parameter set
            job = self.project.open_job(params)
            job.init()
            job.doc.setdefault('status', 'initialized')
            job.doc.setdefault('simulations', [])
            jobs_created += 1

        print(f"Created {jobs_created} simulation jobs")


    def run_job(self, job_id: str) -> str:
        """
        Run simulations for a specific job.
        
        Parameters
        ----------
        job_id : str
            ID of the job to run
            
        Returns
        -------
        str
            Job ID of the completed job
            
        Raises
        ------
        Exception
            If simulation fails
        """
        try:
            job = self.project.open_job(id=job_id)
            params = job.sp  # Get parameters from job statepoint

            # Run multiple simulations with different seed
            results = []
            for i in range(self.n_simulations):
                # Generate deterministic seed based on job ID and simulation index
                seed = hash(f"{job.id}_{i}") % 2**32

                # Create and run simulator
                simulator = LongShortSimulation(params)
                sim_result = simulator.run(seed=seed)

                # Store only metrics to save space
                results.append(sim_result['metrics'])

            # Store result in job document
            job.doc['simulations'] = results
            job.doc['status'] = 'completed'
            return job.id
        
        except Exception as e:
            # Log the error and re-raise
            print(f"Error in job {job_id}: {str(e)}")
            raise
    

    def run_all_jobs(self, max_workers: Optional[int] = None) -> None:
        """
        Run all initialized jobs in parallel.
        
        Parameters
        ----------
        max_workers : int, optional
            Maximum number of worker threads. If None, uses default based on system.
        """
        # Find all jobs that are initialized but not yet completed
        jobs = list(self.project.find_jobs({'doc.status': 'initialized'}))
        job_ids = [job.id for job in jobs]

        if not job_ids:
            print("No jobs to run. Use setup_jobs() first.")
            return
        
        print(f"Running {len(job_ids)} jobs with {max_workers if max_workers else 'default'} workers")

        # Run jobs in parallel using ThreadPoolExecutor
        completed = 0
        failed = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.run_job, job_id) for job_id in job_ids]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    job_id = future.result()
                    completed += 1
                    print(f"Completed job {job_id} ({completed}/{len(job_ids)})")
                except Exception as e:
                    failed += 1
                    print(f"Job failed with error: {e}")

        print(f"Simulation complete. {completed} jobs succeeded, {failed} jobs failed.")

    
    def get_results(self, metric_name: str = 'sharpe_ratio', strategy: str = 'options_overlay') -> Dict[str, Dict[Any, List[float]]]:
        """
        Get aggregated results for a specific metric.
        
        Parameters
        ----------
        metric_name : str, default='sharpe_ratio'
            Name of the metric to extract
        strategy : str, default='options_overlay'
            Strategy to extract metrics from ('long_short' or 'options_overlay')
            
        Returns
        -------
        Dict[str, Dict[Any, List[float]]]
            Dictionary mapping parameter names to dictionaries of parameter values and metric lists
        """
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
                                metric_value = sim[strategy][metric_name]
                                # Skip NaN values
                                if not np.isnan(metric_value):
                                    metric_by_param[value].append(metric_value)

            results[param] = metric_by_param

        return results