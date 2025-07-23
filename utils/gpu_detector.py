import subprocess
import platform
import cv2
import numpy as np
from typing import Dict, List, Tuple

class GPUDetector:
    """Detect and configure GPU acceleration for video processing"""
    
    def __init__(self):
        self.gpu_info = self.detect_gpu()
        self.opencv_info = self.detect_opencv_capabilities()
        self.ffmpeg_info = self.detect_ffmpeg_capabilities()
        
    def detect_gpu(self) -> Dict:
        """Detect available GPU hardware"""
        gpu_info = {
            'nvidia': False,
            'amd': False,
            'intel': False,
            'gpu_names': [],
            'cuda_available': False,
            'opencl_available': False
        }
        
        try:
            # Windows GPU detection
            if platform.system() == "Windows":
                # Try wmic for GPU detection
                try:
                    result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'], 
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        lines = result.stdout.strip().split('\n')
                        for line in lines[1:]:  # Skip header
                            gpu_name = line.strip()
                            if gpu_name and gpu_name != 'Name':
                                gpu_info['gpu_names'].append(gpu_name)
                                
                                # Check GPU types
                                gpu_lower = gpu_name.lower()
                                if 'nvidia' in gpu_lower or 'geforce' in gpu_lower or 'quadro' in gpu_lower:
                                    gpu_info['nvidia'] = True
                                elif 'amd' in gpu_lower or 'radeon' in gpu_lower:
                                    gpu_info['amd'] = True
                                elif 'intel' in gpu_lower or 'uhd' in gpu_lower or 'iris' in gpu_lower:
                                    gpu_info['intel'] = True
                except:
                    pass
            
            # Linux GPU detection
            elif platform.system() == "Linux":
                try:
                    # Try lspci for GPU detection
                    result = subprocess.run(['lspci', '-nn'], capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        for line in result.stdout.split('\n'):
                            if 'VGA' in line or 'Display' in line:
                                gpu_info['gpu_names'].append(line.strip())
                                line_lower = line.lower()
                                if 'nvidia' in line_lower:
                                    gpu_info['nvidia'] = True
                                elif 'amd' in line_lower or 'ati' in line_lower:
                                    gpu_info['amd'] = True
                                elif 'intel' in line_lower:
                                    gpu_info['intel'] = True
                except:
                    pass
            
            # Check CUDA availability
            try:
                result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    gpu_info['cuda_available'] = True
            except:
                pass
            
            # Check OpenCL availability
            try:
                # Try to create OpenCL context
                if cv2.ocl.haveOpenCL():
                    gpu_info['opencl_available'] = True
            except:
                pass
                
        except Exception as e:
            print(f"GPU detection error: {e}")
        
        return gpu_info
    
    def detect_opencv_capabilities(self) -> Dict:
        """Detect OpenCV capabilities"""
        opencv_info = {
            'version': cv2.__version__,
            'opencl_support': False,
            'cuda_support': False,
            'opencl_device_count': 0,
            'opencl_devices': []
        }
        
        try:
            # Check OpenCL support
            if cv2.ocl.haveOpenCL():
                opencv_info['opencl_support'] = True
                
                # Get OpenCL device info
                try:
                    cv2.ocl.setUseOpenCL(True)
                    context = cv2.ocl.Context_create()
                    if context:
                        opencv_info['opencl_device_count'] = context.ndevices()
                        for i in range(context.ndevices()):
                            device = context.device(i)
                            opencv_info['opencl_devices'].append({
                                'name': device.name(),
                                'type': device.type(),
                                'vendor': device.vendorName()
                            })
                except:
                    pass
            
            # Check CUDA support (if available)
            try:
                if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    opencv_info['cuda_support'] = True
            except:
                pass
                
        except Exception as e:
            print(f"OpenCV capability detection error: {e}")
        
        return opencv_info
    
    def detect_ffmpeg_capabilities(self) -> Dict:
        """Detect FFmpeg hardware acceleration capabilities"""
        ffmpeg_info = {
            'available': False,
            'version': '',
            'hardware_encoders': [],
            'hardware_decoders': [],
            'supported_formats': []
        }
        
        try:
            # Check FFmpeg availability
            result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                ffmpeg_info['available'] = True
                lines = result.stdout.split('\n')
                if lines:
                    ffmpeg_info['version'] = lines[0]
            
            # Get hardware encoders
            try:
                result = subprocess.run(['ffmpeg', '-encoders'], capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    hw_encoders = []
                    for line in result.stdout.split('\n'):
                        if any(hw in line for hw in ['qsv', 'nvenc', 'amf', 'videotoolbox']):
                            if 'h264' in line or 'hevc' in line or 'av1' in line:
                                encoder_name = line.split()[1] if len(line.split()) > 1 else ''
                                if encoder_name:
                                    hw_encoders.append(encoder_name)
                    ffmpeg_info['hardware_encoders'] = hw_encoders
            except:
                pass
            
            # Get hardware decoders
            try:
                result = subprocess.run(['ffmpeg', '-decoders'], capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    hw_decoders = []
                    for line in result.stdout.split('\n'):
                        if any(hw in line for hw in ['qsv', 'cuvid', 'amf']):
                            if 'h264' in line or 'hevc' in line:
                                decoder_name = line.split()[1] if len(line.split()) > 1 else ''
                                if decoder_name:
                                    hw_decoders.append(decoder_name)
                    ffmpeg_info['hardware_decoders'] = hw_decoders
            except:
                pass
                
        except Exception as e:
            print(f"FFmpeg capability detection error: {e}")
        
        return ffmpeg_info
    
    def get_optimal_encoder_settings(self) -> Dict:
        """Get optimal encoder settings based on available hardware"""
        settings = {
            'codec': 'libx264',  # Default software encoder
            'preset': 'medium',
            'crf': '18',
            'hardware_acceleration': False,
            'additional_params': []
        }
        
        # Intel QSV (Quick Sync Video)
        if (self.gpu_info['intel'] and 
            'h264_qsv' in self.ffmpeg_info['hardware_encoders']):
            settings.update({
                'codec': 'h264_qsv',
                'preset': 'medium',
                'hardware_acceleration': True,
                'additional_params': ['-global_quality', '18']
            })
            
        # NVIDIA NVENC
        elif (self.gpu_info['nvidia'] and 
              'h264_nvenc' in self.ffmpeg_info['hardware_encoders']):
            settings.update({
                'codec': 'h264_nvenc',
                'preset': 'medium',
                'hardware_acceleration': True,
                'additional_params': ['-cq', '18']
            })
            
        # AMD AMF
        elif (self.gpu_info['amd'] and 
              'h264_amf' in self.ffmpeg_info['hardware_encoders']):
            settings.update({
                'codec': 'h264_amf',
                'preset': 'balanced',
                'hardware_acceleration': True,
                'additional_params': ['-qp_i', '18', '-qp_p', '20']
            })
        
        return settings
    
    def enable_opencv_acceleration(self):
        """Enable OpenCV acceleration if available"""
        if self.opencv_info['opencl_support']:
            try:
                cv2.ocl.setUseOpenCL(True)
                print("✅ OpenCL enabled for OpenCV")
                return True
            except:
                print("❌ Failed to enable OpenCL for OpenCV")
                return False
        return False
    
    def print_gpu_info(self):
        """Print detailed GPU and acceleration information"""
        print("\n" + "="*60)
        print("🚀 GPU & ACCELERATION DETECTION")
        print("="*60)
        
        # GPU Hardware
        print("\n🔧 GPU Hardware:")
        if self.gpu_info['gpu_names']:
            for gpu in self.gpu_info['gpu_names']:
                print(f"  📱 {gpu}")
        else:
            print("  ❌ No GPU detected")
        
        # GPU Types
        gpu_types = []
        if self.gpu_info['nvidia']:
            gpu_types.append("🟢 NVIDIA GPU detected")
        if self.gpu_info['intel']:
            gpu_types.append("🔷 Intel GPU detected")
        if self.gpu_info['amd']:
            gpu_types.append("🔴 AMD GPU detected")
        
        if gpu_types:
            for gpu_type in gpu_types:
                print(f"  {gpu_type}")
            print("  ✅ GPU acceleration available")
        else:
            print("  ❌ No supported GPU found")
        
        # OpenCV Info
        print(f"\n🎥 OpenCV {self.opencv_info['version']}:")
        if self.opencv_info['opencl_support']:
            print("  🔷 OpenCV OpenCL support detected")
            if self.opencv_info['opencl_devices']:
                for device in self.opencv_info['opencl_devices']:
                    print(f"    📱 {device['name']} ({device['vendor']})")
        else:
            print("  ❌ OpenCV OpenCL not available")
        
        if self.opencv_info['cuda_support']:
            print("  🟢 OpenCV CUDA support detected")
        
        # FFmpeg Info
        print(f"\n🎬 FFmpeg:")
        if self.ffmpeg_info['available']:
            print(f"  ✅ {self.ffmpeg_info['version']}")
            
            if self.ffmpeg_info['hardware_encoders']:
                encoders_str = ", ".join(self.ffmpeg_info['hardware_encoders'])
                print(f"  🎬 Supported hardware encoders: {encoders_str}")
            else:
                print("  ❌ No hardware encoders found")
                
            if self.ffmpeg_info['hardware_decoders']:
                decoders_str = ", ".join(self.ffmpeg_info['hardware_decoders'])
                print(f"  🎞️ Supported hardware decoders: {decoders_str}")
        else:
            print("  ❌ FFmpeg not available")
        
        # Optimal Settings
        optimal = self.get_optimal_encoder_settings()
        print(f"\n⚡ Optimal Encoder Settings:")
        print(f"  🎯 Codec: {optimal['codec']}")
        print(f"  🎯 Hardware Acceleration: {'✅ Enabled' if optimal['hardware_acceleration'] else '❌ Disabled'}")
        
        print("="*60)