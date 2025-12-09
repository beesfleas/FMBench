from typing import List, Dict, Any, Optional
import logging
from datasets import load_dataset
import numpy as np
from .base import BaseScenario

log = logging.getLogger(__name__)

class M3Scenario(BaseScenario):
    """
    Scenario for M3 Forecasting Competition (Monthly)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset_name = kwargs.get("dataset_name", "monash_tsf")
        self.subset = kwargs.get("subset", "m3_monthly")
        self.prediction_length = kwargs.get("prediction_length", 18) # M3 Monthly usually 18

    def get_tasks(self) -> List[Dict[str, Any]]:
        """
        Load M3 dataset and prepare tasks.
        Each task is a time series history + asking for forecast.
        """
        log.info(f"Loading dataset: {self.dataset_name} / {self.subset}")
        
        ds = None
        # Strategy 1: monash_tsf
        try:
            ds = load_dataset(self.dataset_name, self.subset, split="train", trust_remote_code=True)
        except Exception as e:
            log.warning(f"Failed to load {self.dataset_name} from monash_tsf: {e}")

        # Strategy 2: autogluon/chronos_datasets
        if ds is None:
            try:
                # Fallback to autogluon/chronos_datasets
                # Try to map 'm3_monthly' -> 'monash_m3_monthly' if needed
                subset = self.subset
                if not subset.startswith("monash_") and "m3" in subset:
                    subset = f"monash_{subset}"
                
                log.info(f"Trying fallback: autogluon/chronos_datasets/{subset}")
                ds = load_dataset("autogluon/chronos_datasets", subset, split="train")
            except Exception as e:
                log.warning(f"Failed to load autogluon/chronos_datasets: {e}")

        tasks = []
        limit = self.config.get("limit", 100)
        
        if ds is None:
            log.warning("All dataset loading strategies failed. Generating synthetic data for verification.")
            # Strategy 3: Synthetic Data
            for i in range(limit):
                # Generate synthetic sine wave
                length = self.prediction_length * 3
                t = np.linspace(0, 4*np.pi, length)
                
                # Add random phase/freq logic for variety
                phase = np.random.rand() * 2 * np.pi
                freq = 1.0 + np.random.rand() * 0.5
                series = np.sin(freq * t + phase) + np.random.normal(0, 0.1, length)
                
                task = {
                    "task_id": f"synthetic_{i}",
                    "time_series_data": series[:-self.prediction_length].tolist(),
                    "ground_truth": series[-self.prediction_length:].tolist(),
                    "prediction_length": self.prediction_length,
                    "task_type": "forecasting"
                }
                tasks.append(task)
            return tasks

        # Process loaded dataset
        count = 0
        for entry in ds:
            if limit and count >= limit:
                break
            
            target = entry.get('target', [])
            if len(target) <= self.prediction_length:
                continue

            # Split into context (history) and ground truth (future)
            context = target[:-self.prediction_length]
            ground_truth = target[-self.prediction_length:]

            task = {
                "task_id": f"m3_{count}",
                "time_series_data": context, # List of floats
                "ground_truth": ground_truth, # List of floats
                "prediction_length": self.prediction_length,
                "task_type": "forecasting"
            }
            tasks.append(task)
            count += 1
            
        return tasks

    def evaluate(self, task: Dict[str, Any], model_output: Any) -> Dict[str, Any]:
        """
        Evaluate forecast against ground truth.
        Metrics: sMAPE, MASE (simplified).
        """
        ground_truth = np.array(task["ground_truth"])
        
        # Model output is expected to be the forecast values (list or array)
        if isinstance(model_output, dict) and "forecast" in model_output:
            forecast = np.array(model_output["forecast"])
        elif isinstance(model_output, (list, np.ndarray)):
             forecast = np.array(model_output)
        else:
            log.warning(f"Invalid model output format: {type(model_output)}")
            return {"sMAPE": float('nan')}

        # Ensure shapes match
        if forecast.shape != ground_truth.shape:
             # truncate or pad? for now just warn
             log.warning(f"Shape mismatch: Forecast {forecast.shape} vs GT {ground_truth.shape}")
             min_len = min(len(forecast), len(ground_truth))
             forecast = forecast[:min_len]
             ground_truth = ground_truth[:min_len]

        # sMAPE
        # sMAPE = 200 * mean( |F - A| / (|A| + |F|) )
        numerator = np.abs(forecast - ground_truth)
        denominator = np.abs(ground_truth) + np.abs(forecast)
        # Avoid division by zero
        # Replace 0 denominator with 1 (result 0 anyway) or mask
        mask = denominator > 0
        smape = np.mean(200 * numerator[mask] / denominator[mask]) if np.any(mask) else 0.0
        
        return {
            "sMAPE": smape,
            "forecast_length": len(forecast)
        }

class GiftEvalScenario(BaseScenario):
    """
    Scenario for GIFT-EVAL dataset (Salesforce/GiftEval).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset_name = kwargs.get("dataset_name", "Salesforce/GiftEval")
        self.subset = kwargs.get("subset", "default")
        self.prediction_length = kwargs.get("prediction_length", 24)

    def get_tasks(self) -> List[Dict[str, Any]]:
        log.info(f"Loading dataset: {self.dataset_name} / {self.subset}")
        try:
            ds = load_dataset(self.dataset_name, self.subset, split="train", trust_remote_code=True, streaming=True)
        except Exception as e:
            log.error(f"Failed to load {self.dataset_name}: {e}")
            return []

        tasks = []
        limit = self.config.get("limit", 100)
        count = 0

        for entry in ds:
            if limit and count >= limit:
                break
            
            # GIFT-EVAL structure: 'target' is the time series
            full_series = entry.get('target', [])
            if not isinstance(full_series, list):
                 # Convert numpy array or other iterables if needed
                 full_series = list(full_series)

            if len(full_series) <= self.prediction_length:
                continue

            context = full_series[:-self.prediction_length]
            ground_truth = full_series[-self.prediction_length:]

            task = {
                "task_id": f"gift_eval_{count}",
                "time_series_data": context,
                "ground_truth": ground_truth,
                "prediction_length": self.prediction_length,
                "task_type": "forecasting"
            }
            tasks.append(task)
            count += 1
            
        return tasks

    def evaluate(self, task: Dict[str, Any], model_output: Any) -> Dict[str, Any]:
        # Reuse simple evaluation logic for now, or inherit from a common base if refactored
        # Copy-pasting logic from M3Scenario.evaluate to be self-contained for now
        ground_truth = np.array(task["ground_truth"])
        
        if isinstance(model_output, dict) and "forecast" in model_output:
            forecast = np.array(model_output["forecast"])
        elif isinstance(model_output, (list, np.ndarray)):
             forecast = np.array(model_output)
        else:
            return {"sMAPE": float('nan')}

        if forecast.shape != ground_truth.shape:
             min_len = min(len(forecast), len(ground_truth))
             forecast = forecast[:min_len]
             ground_truth = ground_truth[:min_len]

        numerator = np.abs(forecast - ground_truth)
        denominator = np.abs(ground_truth) + np.abs(forecast)
        mask = denominator > 0
        smape = np.mean(200 * numerator[mask] / denominator[mask]) if np.any(mask) else 0.0
        
        return {
            "sMAPE": smape,
            "forecast_length": len(forecast)
        }

class FevBenchScenario(BaseScenario):
    """
    Scenario for FEV-Bench (autogluon/fev_datasets).
    Assumes row-based format for some subsets (like ETT) and aggregates by ID.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset_name = kwargs.get("dataset_name", "autogluon/fev_datasets")
        self.subset = kwargs.get("subset", "ETT_1H")
        self.prediction_length = kwargs.get("prediction_length", 24)
        self.id_column = kwargs.get("id_column", "item_id") # May vary by subset, default attempt
        self.target_column = kwargs.get("target_column", "OT") # Default for ETT

    def get_tasks(self) -> List[Dict[str, Any]]:
        log.info(f"Loading dataset: {self.dataset_name} / {self.subset}")
        try:
            ds = load_dataset(self.dataset_name, self.subset, split="train", streaming=True)
        except Exception as e:
            log.error(f"Failed to load {self.dataset_name}: {e}")
            return []

        tasks = []
        limit = self.config.get("limit", 100)
        # Check if dataset is row-based (has 'timestamp' and keys suggesting scalar values per row)
        # For streaming dataset, we cannot index ds[0]. Use iterator.
        try:
            iterator = iter(ds)
            sample = next(iterator)
        except StopIteration:
            log.warning(f"Dataset {self.dataset_name} is empty.")
            return []
            
        is_row_based = "timestamp" in sample and isinstance(sample.get(self.target_column), (int, float))
        
        if is_row_based:
            # Need to aggregate. 
            # Since we consumed one sample, we need to be careful. 
            # If streaming, we might need to reload or chain the sample back.
            # Simpler approach: converts to pandas.
            # For IterableDataset, ds.to_pandas() is not available. We must consume it.
            
            import pandas as pd
            log.info("Detailed loading for row-based dataset (consuming stream)...")
            
            # Recreate chain with the peeked sample
            from itertools import chain
            full_iter = chain([sample], iterator)
            
            # Load all into dataframe (assuming it fits in memory, ETT usually does)
            data_list = list(full_iter) 
            df = pd.DataFrame(data_list)
            
            target_series = df[self.target_column].values
            
            total_len = len(target_series)
            window_size = self.prediction_length * 3 
            step = window_size
            
            count = 0
            for i in range(0, total_len - window_size, step):
                if limit and count >= limit:
                    break
                
                segment = target_series[i : i + window_size]
                context = segment[:-self.prediction_length].tolist()
                ground_truth = segment[-self.prediction_length:].tolist()
                
                tasks.append({
                    "task_id": f"fev_ett_{count}",
                    "time_series_data": context,
                    "ground_truth": ground_truth,
                    "prediction_length": self.prediction_length,
                    "task_type": "forecasting"
                })
                count += 1
                
        else:
            # Standard "series per row" format
            # We already consumed 'sample', so handle it
            
            def process_entry(entry, idx):
                full_series = entry.get(self.target_column, entry.get('target', []))
                # If target is not list, cast it
                if not isinstance(full_series, list):
                     # If numpy or other, try list
                     try:
                         full_series = list(full_series)
                     except:
                         log.warning(f"Failed to convert target to list: {full_series}")
                         pass

                if len(full_series) <= self.prediction_length:
                    return None

                context = full_series[:-self.prediction_length]
                ground_truth = full_series[-self.prediction_length:]

                return {
                    "task_id": f"fev_{idx}",
                    "time_series_data": context,
                    "ground_truth": ground_truth,
                    "prediction_length": self.prediction_length,
                    "task_type": "forecasting"
                }

            # Process the peeked sample
            res = process_entry(sample, 0)
            if res:
                tasks.append(res)
                
            count = 1
            # Continue with iterator
            for entry in iterator:
                if limit and count >= limit:
                    break
                
                res = process_entry(entry, count)
                if res:
                    tasks.append(res)
                count += 1

        return tasks

    def evaluate(self, task: Dict[str, Any], model_output: Any) -> Dict[str, Any]:
        # Same eval logic
        ground_truth = np.array(task["ground_truth"])
        
        if isinstance(model_output, dict) and "forecast" in model_output:
            forecast = np.array(model_output["forecast"])
        elif isinstance(model_output, (list, np.ndarray)):
             forecast = np.array(model_output)
        else:
            return {"sMAPE": float('nan')}

        if forecast.shape != ground_truth.shape:
             min_len = min(len(forecast), len(ground_truth))
             forecast = forecast[:min_len]
             ground_truth = ground_truth[:min_len]

        numerator = np.abs(forecast - ground_truth)
        denominator = np.abs(ground_truth) + np.abs(forecast)
        mask = denominator > 0
        smape = np.mean(200 * numerator[mask] / denominator[mask]) if np.any(mask) else 0.0
        
        return {
            "sMAPE": smape,
            "forecast_length": len(forecast)
        }
