import time
import subprocess
import threading

class PowerMetrics:
    def __init__(self):
        self.gpu_power = 0
        self.cpu_power = 0
        self.gpu_frequency = 0
        self.gpu_utilization = 0
        self.monitoring = False
        self.thread = None
        self.sudo_available = self._check_sudo_available()
    
    def _check_sudo_available(self) -> bool:
        """Check if sudo powermetrics is available without password"""
        try:
            result = subprocess.run([
                'sudo', '-n', 'powermetrics', '--help'
            ], capture_output=True, text=True, timeout=1)
            return result.returncode == 0
        except (subprocess.SubprocessError, OSError, FileNotFoundError):
            return False
    
    def start_monitoring(self):
        """Start background power monitoring if available"""
        if self.sudo_available and not self.monitoring:
            self.monitoring = True
            self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.thread.start()
            print("🔋 Power monitoring started")
        else:
            print("⚠️  Power monitoring disabled (sudo powermetrics not available)")
    
    def stop_monitoring(self):
        """Stop background power monitoring"""
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=1)
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                # Quick powermetrics sample without password prompt
                result = subprocess.run([
                    'sudo', '-n', 'powermetrics', 
                    '--samplers', 'gpu_power,cpu_power',
                    '--sample-count', '1',
                    '-n', '1'
                ], capture_output=True, text=True, timeout=2)
                
                if result.returncode == 0:
                    self._parse_powermetrics(result.stdout)
                    
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                pass
            except OSError:
                pass

            time.sleep(1.0)  # Update every second
    
    def _parse_powermetrics(self, output: str):
        """Parse powermetrics output"""
        try:
            lines = output.split('\n')
            for line in lines:
                if 'GPU Power:' in line:
                    power_str = line.split('GPU Power:')[1].strip()
                    if 'mW' in power_str:
                        self.gpu_power = int(power_str.split()[0])
                elif 'CPU Power:' in line:
                    power_str = line.split('CPU Power:')[1].strip()
                    if 'mW' in power_str:
                        self.cpu_power = int(power_str.split()[0])
                elif 'GPU HW active frequency:' in line:
                    freq_str = line.split('GPU HW active frequency:')[1].strip()
                    if 'MHz' in freq_str:
                        self.gpu_frequency = int(freq_str.split()[0])
                elif 'GPU HW active residency:' in line:
                    util_str = line.split('GPU HW active residency:')[1].strip()
                    if '%' in util_str:
                        self.gpu_utilization = float(util_str.split('%')[0])
        except (ValueError, IndexError, AttributeError):
            pass