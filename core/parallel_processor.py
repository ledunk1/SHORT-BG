import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from typing import List, Tuple, Callable, Optional
import threading
import queue
import time
from pathlib import Path
import gc

class ParallelFrameProcessor:
    """Parallel frame processor using multi-threading and multi-processing"""
    
    def __init__(self, max_workers: Optional[int] = None, use_gpu: bool = True):
        """
        Initialize parallel processor
        
        Args:
            max_workers: Maximum number of worker threads/processes
            use_gpu: Whether to use GPU acceleration
        """
        self.cpu_count = mp.cpu_count()
        self.max_workers = max_workers or min(self.cpu_count, 16)  # Limit to 16 for memory
        self.use_gpu = use_gpu
        
        # GPU acceleration setup
        if self.use_gpu:
            self.setup_gpu_acceleration()
        
        print(f"🚀 Parallel Processor initialized:")
        print(f"  💻 CPU cores: {self.cpu_count}")
        print(f"  🔧 Max workers: {self.max_workers}")
        print(f"  ⚡ GPU acceleration: {'✅ Enabled' if self.use_gpu else '❌ Disabled'}")
    
    def setup_gpu_acceleration(self):
        """Setup GPU acceleration for OpenCV"""
        try:
            if cv2.ocl.haveOpenCL():
                cv2.ocl.setUseOpenCL(True)
                print("  🔷 OpenCL enabled for parallel processing")
            else:
                print("  ❌ OpenCL not available")
                self.use_gpu = False
        except Exception as e:
            print(f"  ⚠️ GPU setup warning: {e}")
            self.use_gpu = False
    
    def process_frames_parallel(self, frames_data: List[Tuple], 
                               process_function: Callable,
                               progress_callback: Optional[Callable] = None,
                               chunk_size: int = 4) -> List:
        """
        Process frames in parallel using thread pool
        
        Args:
            frames_data: List of frame data tuples
            process_function: Function to process each frame
            progress_callback: Progress callback function
            chunk_size: Number of frames to process per chunk
            
        Returns:
            List of processed results
        """
        total_frames = len(frames_data)
        results = [None] * total_frames
        completed = 0
        
        print(f"🎬 Processing {total_frames} frames with {self.max_workers} workers...")
        start_time = time.time()
        
        # Create chunks for better memory management
        chunks = [frames_data[i:i + chunk_size] for i in range(0, len(frames_data), chunk_size)]
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit chunk processing tasks
            future_to_chunk = {}
            
            for chunk_idx, chunk in enumerate(chunks):
                future = executor.submit(self._process_chunk, chunk, process_function, chunk_idx * chunk_size)
                future_to_chunk[future] = (chunk_idx, chunk)
            
            # Collect results as they complete
            for future in as_completed(future_to_chunk):
                chunk_idx, chunk = future_to_chunk[future]
                start_idx = chunk_idx * chunk_size
                
                try:
                    chunk_results = future.result()
                    
                    # Store results in correct positions
                    for i, result in enumerate(chunk_results):
                        results[start_idx + i] = result
                    
                    completed += len(chunk_results)
                    
                    # Progress callback
                    if progress_callback:
                        progress_callback(completed / total_frames)
                    
                    # Progress logging
                    elapsed = time.time() - start_time
                    fps = completed / elapsed if elapsed > 0 else 0
                    eta = (total_frames - completed) / fps if fps > 0 else 0
                    
                    print(f"  🎯 Progress: {completed}/{total_frames} "
                          f"({completed/total_frames*100:.1f}%) "
                          f"| Speed: {fps:.1f} fps | ETA: {eta:.1f}s")
                    
                except Exception as e:
                    print(f"  ❌ Chunk processing error: {e}")
                    # Fill with None for failed chunk
                    for i in range(len(chunk)):
                        results[start_idx + i] = None
        
        processing_time = time.time() - start_time
        avg_fps = total_frames / processing_time if processing_time > 0 else 0
        
        print(f"✅ Parallel processing complete!")
        print(f"  ⏱️ Total time: {processing_time:.2f}s")
        print(f"  📊 Average speed: {avg_fps:.1f} fps")
        print(f"  🚀 Speedup: ~{self.max_workers}x theoretical")
        
        return results
    
    def _process_chunk(self, chunk: List[Tuple], process_function: Callable, start_idx: int) -> List:
        """
        Process a chunk of frames in a single thread
        
        Args:
            chunk: Chunk of frame data
            process_function: Processing function
            start_idx: Starting index for this chunk
            
        Returns:
            List of processed results for this chunk
        """
        results = []
        
        for i, frame_data in enumerate(chunk):
            try:
                # Add chunk info to frame data
                frame_idx = start_idx + i
                result = process_function(frame_data, frame_idx)
                results.append(result)
            except Exception as e:
                print(f"    ⚠️ Frame {start_idx + i} processing error: {e}")
                results.append(None)
        
        return results
    
    def process_frames_pipeline(self, frames_data: List[Tuple],
                               process_function: Callable,
                               progress_callback: Optional[Callable] = None,
                               buffer_size: int = 8) -> List:
        """
        Process frames using streaming pipeline approach for minimal RAM usage
        
        Args:
            frames_data: List of frame data tuples
            process_function: Function to process each frame
            progress_callback: Progress callback function
            buffer_size: Size of processing buffer (reduced for RAM efficiency)
            
        Returns:
            List of processed results
        """
        total_frames = len(frames_data)
        
        print(f"🔄 Streaming pipeline processing {total_frames} frames (RAM efficient)...")
        start_time = time.time()
        
        # Create smaller queues for streaming
        input_queue = queue.Queue(maxsize=buffer_size)
        output_queue = queue.Queue(maxsize=buffer_size)
        
        # Results will be yielded as they're ready (streaming)
        results = []
        
        # Producer thread - feeds frames to workers
        def producer():
            for i, frame_data in enumerate(frames_data):
                input_queue.put((i, frame_data))
            
            # Signal end of input
            for _ in range(self.max_workers):
                input_queue.put(None)
        
        # Worker threads - process frames with memory cleanup
        def worker():
            while True:
                item = input_queue.get()
                if item is None:
                    break
                
                frame_idx, frame_data = item
                try:
                    result = process_function(frame_data, frame_idx)
                    output_queue.put((frame_idx, result))
                    
                    # Force garbage collection periodically
                    if frame_idx % 10 == 0:
                        gc.collect()
                        
                except Exception as e:
                    print(f"    ⚠️ Frame {frame_idx} processing error: {e}")
                    output_queue.put((frame_idx, None))
                finally:
                    input_queue.task_done()
        
        # Start producer thread
        producer_thread = threading.Thread(target=producer)
        producer_thread.start()
        
        # Start worker threads (reduced count for RAM efficiency)
        worker_threads = []
        worker_count = min(self.max_workers, 4)  # Limit workers to save RAM
        for _ in range(worker_count):
            thread = threading.Thread(target=worker)
            thread.start()
            worker_threads.append(thread)
        
        # Collect results in order and store temporarily
        result_buffer = {}
        completed = 0
        next_expected = 0
        
        while completed < total_frames:
            try:
                frame_idx, result = output_queue.get(timeout=30)
                result_buffer[frame_idx] = result
                completed += 1
                
                # Add results to final list in order
                while next_expected in result_buffer:
                    results.append(result_buffer.pop(next_expected))
                    next_expected += 1
                    
                    # Clear memory periodically
                    if len(results) % 20 == 0:
                        gc.collect()
                
                # Progress callback
                if progress_callback:
                    progress_callback(completed / total_frames)
                
                # Progress logging
                if completed % 20 == 0 or completed == total_frames:
                    elapsed = time.time() - start_time
                    fps = completed / elapsed if elapsed > 0 else 0
                    eta = (total_frames - completed) / fps if fps > 0 else 0
                    
                    print(f"  🔄 Streaming: {completed}/{total_frames} "
                          f"({completed/total_frames*100:.1f}%) "
                          f"| Speed: {fps:.1f} fps | ETA: {eta:.1f}s")
                
            except queue.Empty:
                print("  ⚠️ Streaming timeout - continuing...")
                break
        
        # Add any remaining results
        while next_expected < total_frames and next_expected in result_buffer:
            results.append(result_buffer.pop(next_expected))
            next_expected += 1
        
        # Wait for all threads to complete
        producer_thread.join()
        for thread in worker_threads:
            thread.join()
        
        # Final cleanup
        del result_buffer
        gc.collect()
        
        processing_time = time.time() - start_time
        avg_fps = total_frames / processing_time if processing_time > 0 else 0
        
        print(f"✅ Streaming pipeline complete!")
        print(f"  ⏱️ Total time: {processing_time:.2f}s")
        print(f"  📊 Average speed: {avg_fps:.1f} fps")
        print(f"  💾 RAM efficient: {worker_count} workers, {buffer_size} buffer")
        
        return results


class GPUAcceleratedProcessor:
    """GPU-accelerated frame processing using OpenCV"""
    
    def __init__(self):
        self.gpu_available = self.setup_gpu()
    
    def setup_gpu(self) -> bool:
        """Setup GPU acceleration"""
        try:
            if cv2.ocl.haveOpenCL():
                cv2.ocl.setUseOpenCL(True)
                print("🔷 GPU acceleration enabled (OpenCL)")
                return True
            else:
                print("❌ GPU acceleration not available")
                return False
        except Exception as e:
            print(f"⚠️ GPU setup error: {e}")
            return False
    
    def resize_gpu(self, image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """GPU-accelerated image resizing"""
        if not self.gpu_available:
            return cv2.resize(image, size, interpolation=cv2.INTER_LANCZOS4)
        
        try:
            # Upload to GPU
            gpu_img = cv2.UMat(image)
            
            # Resize on GPU
            gpu_resized = cv2.resize(gpu_img, size, interpolation=cv2.INTER_LANCZOS4)
            
            # Download from GPU
            return gpu_resized.get()
        except:
            # Fallback to CPU
            return cv2.resize(image, size, interpolation=cv2.INTER_LANCZOS4)
    
    def blend_gpu(self, background: np.ndarray, foreground: np.ndarray, 
                  mask: np.ndarray) -> np.ndarray:
        """GPU-accelerated image blending"""
        if not self.gpu_available:
            return self._blend_cpu(background, foreground, mask)
        
        try:
            # Upload to GPU
            gpu_bg = cv2.UMat(background)
            gpu_fg = cv2.UMat(foreground)
            gpu_mask = cv2.UMat(mask)
            
            # Normalize mask
            gpu_mask_norm = gpu_mask.astype(np.float32) / 255.0
            if len(gpu_mask_norm.shape) == 2:
                gpu_mask_norm = cv2.merge([gpu_mask_norm, gpu_mask_norm, gpu_mask_norm])
            
            # Blend on GPU
            gpu_bg_float = gpu_bg.astype(np.float32)
            gpu_fg_float = gpu_fg.astype(np.float32)
            
            gpu_result = gpu_bg_float * (1 - gpu_mask_norm) + gpu_fg_float * gpu_mask_norm
            
            # Download from GPU
            return gpu_result.get().astype(np.uint8)
        except:
            # Fallback to CPU
            return self._blend_cpu(background, foreground, mask)
    
    def _blend_cpu(self, background: np.ndarray, foreground: np.ndarray, 
                   mask: np.ndarray) -> np.ndarray:
        """CPU fallback for blending"""
        mask_norm = mask.astype(np.float32) / 255.0
        if len(mask_norm.shape) == 2:
            mask_norm = cv2.merge([mask_norm, mask_norm, mask_norm])
        
        bg_float = background.astype(np.float32)
        fg_float = foreground.astype(np.float32)
        
        result = bg_float * (1 - mask_norm) + fg_float * mask_norm
        return result.astype(np.uint8)
    
    def morphology_gpu(self, image: np.ndarray, operation: int, 
                       kernel_size: int = 5) -> np.ndarray:
        """GPU-accelerated morphological operations"""
        if not self.gpu_available:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            return cv2.morphologyEx(image, operation, kernel)
        
        try:
            # Upload to GPU
            gpu_img = cv2.UMat(image)
            
            # Create kernel
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            
            # Morphological operation on GPU
            gpu_result = cv2.morphologyEx(gpu_img, operation, kernel)
            
            # Download from GPU
            return gpu_result.get()
        except:
            # Fallback to CPU
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            return cv2.morphologyEx(image, operation, kernel)
    
    def gaussian_blur_gpu(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """GPU-accelerated Gaussian blur"""
        if not self.gpu_available:
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        try:
            # Upload to GPU
            gpu_img = cv2.UMat(image)
            
            # Blur on GPU
            gpu_result = cv2.GaussianBlur(gpu_img, (kernel_size, kernel_size), 0)
            
            # Download from GPU
            return gpu_result.get()
        except:
            # Fallback to CPU
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)