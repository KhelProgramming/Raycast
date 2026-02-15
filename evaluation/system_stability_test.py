import time
import threading
import queue
import numpy as np
import pandas as pd
from tabulate import tabulate

# --- ACTUAL PROJECT IMPORTS ---
from gesture_system.ml.trainer import Trainer
from gesture_system.ml.predictor import Predictor
from gesture_system.config import TRAIN_FILE

class SystemStabilityTest:
    def __init__(self):
        # 1. Component Initialization
        trainer = Trainer(TRAIN_FILE)
        model, scaler = trainer.train()
        self.predictor = Predictor(model, scaler)
        
        # 2. Pruned Dataset Loading (N=4,417)
        df = pd.read_csv(TRAIN_FILE)
        self.test_data = df.iloc[:, :-1].values 
        
        # 3. System Constraints
        self.frame_queue = queue.Queue(maxsize=2) # REAL-TIME PRESSURE TEST
        self.results = []
        self.frame_loss_count = 0 # Professional Terminology
        self.stop_event = threading.Event()

    def producer_thread(self):
        """STAGE 1: Deterministic Camera Metronome"""
        TARGET = 1/30
        next_frame_time = time.perf_counter()
        print(f"⏱️ Metronome started at {1/TARGET} FPS...")
        
        for row in self.test_data:
            if self.stop_event.is_set(): break
            
            # The Metronome Tick
            next_frame_time += TARGET
            
            try:
                # Non-blocking: We'd rather lose a frame than lag the system
                self.frame_queue.put((row, time.perf_counter()), block=False)
            except queue.Full:
                self.frame_loss_count += 1
            
            # Precise sleep calculation
            sleep_time = next_frame_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
                
        self.stop_event.set()

    def consumer_thread(self):
        """STAGE 2: The Worker (Compute & Action)"""
        while not self.stop_event.is_set() or not self.frame_queue.empty():
            try:
                row, entry_time = self.frame_queue.get(timeout=1)
            except queue.Empty: break

            # --- WORKLOAD ---
            start_compute = time.perf_counter()
            _ = self.predictor.predict(row) # Real ML Work
            time.sleep(0.002) # Fixed OS Action Overhead
            end_compute = time.perf_counter()
            
            # --- METRICS ---
            self.results.append({
                "latency": (end_compute - entry_time) * 1000,
                "compute": (end_compute - start_compute) * 1000,
                "backlog": self.frame_queue.qsize()
            })
            self.frame_queue.task_done()

    def run_defense_benchmark(self):
        t1 = threading.Thread(target=self.producer_thread)
        t2 = threading.Thread(target=self.consumer_thread)
        
        start_bench = time.perf_counter()
        t1.start(); t2.start()
        t1.join(); t2.join()
        total_time = time.perf_counter() - start_bench

        # --- STATISTICAL ANALYSIS ---
        latencies = [r['latency'] for r in self.results]
        computes = [r['compute'] for r in self.results]
        
        # P95 and Worst-case show the 'Lag Spikes'
        p95_lat = np.percentile(latencies, 95)
        worst_lat = np.max(latencies)
        avg_fps = len(self.results) / total_time
        frame_loss_rate = (self.frame_loss_count / len(self.test_data)) * 100
        peak_backlog = max(r['backlog'] for r in self.results)

        # Final Report Data
        report = [[
            f"{avg_fps:.2f}",
            f"{np.mean(latencies):.2f}ms",
            f"{p95_lat:.2f}ms", # The "Truth" Metric
            f"{worst_lat:.2f}ms",
            f"{frame_loss_rate:.2f}%",
            f"{peak_backlog}/2",
            "STABLE" if p95_lat < 33.3 else "🔴 UNSTABLE"
        ]]
        
        print("\n" + "="*105)
        print("🏆 DETERMINISTIC SYSTEM STABILITY REPORT (N=4,417)")
        print("="*105)
        headers = ["Actual FPS", "Avg Latency", "P95 Latency", "Worst Case", "Frame Loss", "Peak Backlog", "Budget Status"]
        print(tabulate(report, headers=headers, tablefmt="fancy_grid"))

if __name__ == "__main__":
    benchmark = SystemStabilityTest()
    benchmark.run_defense_benchmark()
