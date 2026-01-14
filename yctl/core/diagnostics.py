"""
System diagnostics for checking AI development environment.
"""

import sys
import subprocess
import platform
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DiagnosticResult:
    """Result of a diagnostic check."""
    name: str
    status: str  # 'ok', 'warning', 'error'
    message: str
    details: Optional[str] = None
    fix_suggestion: Optional[str] = None


class SystemDiagnostics:
    """Perform system diagnostics for AI development."""
    
    def __init__(self):
        self.results: List[DiagnosticResult] = []
    
    def run_all_checks(self) -> List[DiagnosticResult]:
        """Run all diagnostic checks."""
        self.check_python_version()
        self.check_pip()
        self.check_venv()
        self.check_gpu()
        self.check_cuda()
        self.check_common_tools()
        return self.results
    
    def check_python_version(self) -> None:
        """Check Python version."""
        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"
        
        if version.major == 3 and version.minor >= 10:
            self.results.append(DiagnosticResult(
                name="Python Version",
                status="ok",
                message=f"Python {version_str}",
                details="Python version is compatible"
            ))
        elif version.major == 3 and version.minor >= 8:
            self.results.append(DiagnosticResult(
                name="Python Version",
                status="warning",
                message=f"Python {version_str}",
                details="Python 3.10+ is recommended for best compatibility",
                fix_suggestion="Consider upgrading: sudo apt install python3.10"
            ))
        else:
            self.results.append(DiagnosticResult(
                name="Python Version",
                status="error",
                message=f"Python {version_str}",
                details="Python 3.10+ is required",
                fix_suggestion="Upgrade Python: sudo apt install python3.10"
            ))
    
    def check_pip(self) -> None:
        """Check pip installation and version."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                pip_version = result.stdout.strip()
                self.results.append(DiagnosticResult(
                    name="pip",
                    status="ok",
                    message="pip is installed",
                    details=pip_version
                ))
            else:
                self.results.append(DiagnosticResult(
                    name="pip",
                    status="error",
                    message="pip check failed",
                    details=result.stderr,
                    fix_suggestion="Reinstall pip: python3 -m ensurepip --upgrade"
                ))
        except Exception as e:
            self.results.append(DiagnosticResult(
                name="pip",
                status="error",
                message="pip not found",
                details=str(e),
                fix_suggestion="Install pip: sudo apt install python3-pip"
            ))
    
    def check_venv(self) -> None:
        """Check if venv module is available."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "venv", "--help"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                self.results.append(DiagnosticResult(
                    name="venv",
                    status="ok",
                    message="venv module is available",
                    details="Can create virtual environments"
                ))
            else:
                self.results.append(DiagnosticResult(
                    name="venv",
                    status="error",
                    message="venv module not working",
                    details=result.stderr,
                    fix_suggestion="Install venv: sudo apt install python3-venv"
                ))
        except Exception as e:
            self.results.append(DiagnosticResult(
                name="venv",
                status="error",
                message="venv module not found",
                details=str(e),
                fix_suggestion="Install venv: sudo apt install python3-venv"
            ))
    
    def check_gpu(self) -> None:
        """Check GPU availability."""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            
            if gpus:
                gpu_info = []
                for gpu in gpus:
                    gpu_info.append(f"{gpu.name} ({gpu.memoryTotal}MB)")
                
                self.results.append(DiagnosticResult(
                    name="GPU",
                    status="ok",
                    message=f"Found {len(gpus)} GPU(s)",
                    details="\n".join(gpu_info)
                ))
            else:
                self.results.append(DiagnosticResult(
                    name="GPU",
                    status="warning",
                    message="No GPU detected",
                    details="Training will use CPU only (slower)",
                    fix_suggestion="Check if NVIDIA drivers are installed: nvidia-smi"
                ))
        except ImportError:
            self.results.append(DiagnosticResult(
                name="GPU",
                status="warning",
                message="GPUtil not installed",
                details="Cannot check GPU status",
                fix_suggestion="Install GPUtil: pip install gputil"
            ))
        except Exception as e:
            self.results.append(DiagnosticResult(
                name="GPU",
                status="warning",
                message="GPU check failed",
                details=str(e),
                fix_suggestion="Check NVIDIA drivers: nvidia-smi"
            ))
    
    def check_cuda(self) -> None:
        """Check CUDA availability."""
        # Check nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                # Extract CUDA version from nvidia-smi output
                lines = result.stdout.split('\n')
                cuda_version = "Unknown"
                for line in lines:
                    if "CUDA Version:" in line:
                        cuda_version = line.split("CUDA Version:")[-1].strip().split()[0]
                        break
                
                self.results.append(DiagnosticResult(
                    name="CUDA",
                    status="ok",
                    message=f"CUDA is available",
                    details=f"CUDA Version: {cuda_version}"
                ))
            else:
                self.results.append(DiagnosticResult(
                    name="CUDA",
                    status="warning",
                    message="nvidia-smi failed",
                    details="CUDA may not be properly configured",
                    fix_suggestion="Install NVIDIA drivers: sudo ubuntu-drivers autoinstall"
                ))
        except FileNotFoundError:
            self.results.append(DiagnosticResult(
                name="CUDA",
                status="warning",
                message="CUDA not detected",
                details="nvidia-smi not found",
                fix_suggestion="Install NVIDIA drivers: sudo ubuntu-drivers autoinstall"
            ))
        except Exception as e:
            self.results.append(DiagnosticResult(
                name="CUDA",
                status="warning",
                message="CUDA check failed",
                details=str(e)
            ))
        
        # Check PyTorch CUDA
        try:
            import torch
            if torch.cuda.is_available():
                cuda_version = torch.version.cuda
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
                
                self.results.append(DiagnosticResult(
                    name="PyTorch CUDA",
                    status="ok",
                    message=f"PyTorch can use CUDA",
                    details=f"CUDA {cuda_version}, {device_count} device(s), {device_name}"
                ))
            else:
                self.results.append(DiagnosticResult(
                    name="PyTorch CUDA",
                    status="warning",
                    message="PyTorch CUDA not available",
                    details="PyTorch is installed but cannot access CUDA",
                    fix_suggestion="Reinstall PyTorch with CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
                ))
        except ImportError:
            self.results.append(DiagnosticResult(
                name="PyTorch CUDA",
                status="warning",
                message="PyTorch not installed",
                details="Cannot check PyTorch CUDA support",
                fix_suggestion="Install PyTorch: pip install torch torchvision"
            ))
    
    def check_common_tools(self) -> None:
        """Check common development tools."""
        tools = {
            "git": "Version control",
            "docker": "Containerization",
            "tmux": "Terminal multiplexer",
            "htop": "Process monitor",
        }
        
        for tool, description in tools.items():
            try:
                result = subprocess.run(
                    ["which", tool],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.returncode == 0:
                    self.results.append(DiagnosticResult(
                        name=tool,
                        status="ok",
                        message=f"{tool} is installed",
                        details=description
                    ))
                else:
                    self.results.append(DiagnosticResult(
                        name=tool,
                        status="warning",
                        message=f"{tool} not found",
                        details=f"Optional: {description}",
                        fix_suggestion=f"Install {tool}: sudo apt install {tool}"
                    ))
            except Exception:
                pass


def run_diagnostics() -> List[DiagnosticResult]:
    """Run all system diagnostics and return results."""
    diagnostics = SystemDiagnostics()
    return diagnostics.run_all_checks()
