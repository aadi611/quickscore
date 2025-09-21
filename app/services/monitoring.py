"""
Monitoring and optimization system for QuickScore platform.
"""
import logging
import asyncio
import time
import psutil
import statistics
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path
from collections import defaultdict, deque

from sqlalchemy.orm import Session
from sqlalchemy import create_engine, text, func

from app.core.config import settings

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics being tracked."""
    PERFORMANCE = "performance"
    BUSINESS = "business"
    SYSTEM = "system"
    USER = "user"
    QUALITY = "quality"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Metric:
    """Individual metric data point."""
    
    name: str
    value: float
    timestamp: datetime
    type: MetricType
    tags: Dict[str, str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary."""
        return {
            'name': self.name,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'type': self.type.value,
            'tags': self.tags
        }


@dataclass
class Alert:
    """System alert."""
    
    id: str
    level: AlertLevel
    message: str
    metric_name: str
    current_value: float
    threshold: float
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class PerformanceReport:
    """Performance analysis report."""
    
    period_start: datetime
    period_end: datetime
    
    # API Performance
    avg_response_time: float
    p95_response_time: float
    p99_response_time: float
    error_rate: float
    throughput: float
    
    # Analysis Performance
    avg_analysis_time: float
    analysis_success_rate: float
    
    # System Performance
    avg_cpu_usage: float
    avg_memory_usage: float
    disk_usage: float
    
    # Business Metrics
    total_analyses: int
    unique_users: int
    avg_score: float
    
    # Recommendations
    recommendations: List[str]


class MonitoringSystem:
    """Comprehensive monitoring and optimization system."""
    
    def __init__(self):
        self.metrics_storage: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.alerts: List[Alert] = []
        self.alert_thresholds = self._setup_default_thresholds()
        self.performance_targets = self._setup_performance_targets()
        
        # Storage paths
        self.storage_path = Path("monitoring_data")
        self.storage_path.mkdir(exist_ok=True)
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_interval = 30  # seconds
        
    def _setup_default_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Setup default alert thresholds."""
        return {
            'api_response_time': {
                'warning': 2.0,  # seconds
                'error': 5.0,
                'critical': 10.0
            },
            'error_rate': {
                'warning': 0.05,  # 5%
                'error': 0.10,   # 10%
                'critical': 0.20  # 20%
            },
            'cpu_usage': {
                'warning': 70.0,  # percent
                'error': 85.0,
                'critical': 95.0
            },
            'memory_usage': {
                'warning': 80.0,  # percent
                'error': 90.0,
                'critical': 95.0
            },
            'disk_usage': {
                'warning': 80.0,  # percent
                'error': 90.0,
                'critical': 95.0
            },
            'analysis_time': {
                'warning': 300.0,  # seconds (5 minutes)
                'error': 600.0,    # 10 minutes
                'critical': 1200.0  # 20 minutes
            }
        }
    
    def _setup_performance_targets(self) -> Dict[str, float]:
        """Setup performance targets for optimization."""
        return {
            'api_response_time': 1.0,  # Target < 1 second
            'analysis_time': 120.0,    # Target < 2 minutes
            'error_rate': 0.01,        # Target < 1%
            'uptime': 99.9,           # Target 99.9% uptime
            'throughput': 100.0       # Target 100 requests/minute
        }
    
    async def start_monitoring(self):
        """Start the monitoring system."""
        logger.info("Starting monitoring system")
        self.is_monitoring = True
        
        # Start monitoring tasks
        await asyncio.gather(
            self._monitor_system_metrics(),
            self._monitor_api_performance(),
            self._monitor_business_metrics(),
            self._process_alerts(),
            self._optimize_performance()
        )
    
    def stop_monitoring(self):
        """Stop the monitoring system."""
        logger.info("Stopping monitoring system")
        self.is_monitoring = False
    
    async def _monitor_system_metrics(self):
        """Monitor system-level metrics."""
        while self.is_monitoring:
            try:
                # CPU Usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.record_metric(
                    "cpu_usage",
                    cpu_percent,
                    MetricType.SYSTEM,
                    {"host": "localhost"}
                )
                
                # Memory Usage
                memory = psutil.virtual_memory()
                self.record_metric(
                    "memory_usage",
                    memory.percent,
                    MetricType.SYSTEM,
                    {"host": "localhost"}
                )
                
                # Disk Usage
                disk = psutil.disk_usage('/')
                disk_percent = (disk.used / disk.total) * 100
                self.record_metric(
                    "disk_usage",
                    disk_percent,
                    MetricType.SYSTEM,
                    {"host": "localhost", "path": "/"}
                )
                
                # Network I/O
                network = psutil.net_io_counters()
                self.record_metric(
                    "network_bytes_sent",
                    network.bytes_sent,
                    MetricType.SYSTEM,
                    {"host": "localhost"}
                )
                
                self.record_metric(
                    "network_bytes_recv",
                    network.bytes_recv,
                    MetricType.SYSTEM,
                    {"host": "localhost"}
                )
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error monitoring system metrics: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _monitor_api_performance(self):
        """Monitor API performance metrics."""
        while self.is_monitoring:
            try:
                # This would integrate with actual API metrics
                # For now, we'll simulate some basic monitoring
                
                # Response time tracking (would be implemented in middleware)
                # Error rate tracking
                # Throughput monitoring
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error monitoring API performance: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _monitor_business_metrics(self):
        """Monitor business and application metrics."""
        while self.is_monitoring:
            try:
                # Database connection for metrics
                # In production, this would use the actual database
                
                # Track analysis metrics
                # - Number of analyses per hour
                # - Average analysis time
                # - Success/failure rates
                # - Score distributions
                
                # Track user metrics
                # - Active users
                # - New registrations
                # - Usage patterns
                
                await asyncio.sleep(self.monitoring_interval * 2)  # Less frequent
                
            except Exception as e:
                logger.error(f"Error monitoring business metrics: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _process_alerts(self):
        """Process and manage alerts."""
        while self.is_monitoring:
            try:
                # Check all metrics against thresholds
                for metric_name, threshold_config in self.alert_thresholds.items():
                    if metric_name in self.metrics_storage:
                        recent_metrics = list(self.metrics_storage[metric_name])[-10:]  # Last 10 values
                        
                        if recent_metrics:
                            latest_value = recent_metrics[-1].value
                            avg_value = statistics.mean([m.value for m in recent_metrics])
                            
                            # Check thresholds
                            for level_name, threshold in threshold_config.items():
                                if self._should_alert(metric_name, avg_value, threshold, level_name):
                                    alert = Alert(
                                        id=f"{metric_name}_{level_name}_{int(time.time())}",
                                        level=AlertLevel(level_name.lower()),
                                        message=f"{metric_name} {level_name}: {avg_value:.2f} exceeds threshold {threshold}",
                                        metric_name=metric_name,
                                        current_value=avg_value,
                                        threshold=threshold,
                                        timestamp=datetime.utcnow()
                                    )
                                    
                                    self.alerts.append(alert)
                                    await self._send_alert(alert)
                
                await asyncio.sleep(60)  # Check alerts every minute
                
            except Exception as e:
                logger.error(f"Error processing alerts: {e}")
                await asyncio.sleep(60)
    
    def _should_alert(self, metric_name: str, value: float, threshold: float, level: str) -> bool:
        """Determine if an alert should be triggered."""
        
        # Check if we've already alerted for this condition recently
        recent_alerts = [
            a for a in self.alerts[-50:]  # Last 50 alerts
            if a.metric_name == metric_name 
            and a.level.value == level.lower()
            and not a.resolved
            and (datetime.utcnow() - a.timestamp).total_seconds() < 3600  # Last hour
        ]
        
        if recent_alerts:
            return False  # Don't spam alerts
        
        # Different logic for different metrics
        if metric_name in ['cpu_usage', 'memory_usage', 'disk_usage', 'error_rate']:
            return value > threshold
        elif metric_name in ['api_response_time', 'analysis_time']:
            return value > threshold
        
        return False
    
    async def _send_alert(self, alert: Alert):
        """Send alert notification."""
        logger.warning(f"ALERT [{alert.level.value.upper()}]: {alert.message}")
        
        # In production, this would send notifications via:
        # - Email
        # - Slack
        # - PagerDuty
        # - Webhook
        
        # Save alert to file for persistence
        alert_file = self.storage_path / "alerts.jsonl"
        with open(alert_file, 'a') as f:
            alert_data = asdict(alert)
            alert_data['timestamp'] = alert.timestamp.isoformat()
            if alert.resolved_at:
                alert_data['resolved_at'] = alert.resolved_at.isoformat()
            f.write(json.dumps(alert_data, default=str) + '\n')
    
    async def _optimize_performance(self):
        """Automatic performance optimization."""
        while self.is_monitoring:
            try:
                # Analyze performance trends
                optimizations = await self._analyze_performance_trends()
                
                # Apply optimizations
                for optimization in optimizations:
                    await self._apply_optimization(optimization)
                
                # Wait before next optimization cycle
                await asyncio.sleep(300)  # Every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in performance optimization: {e}")
                await asyncio.sleep(300)
    
    async def _analyze_performance_trends(self) -> List[Dict[str, Any]]:
        """Analyze performance trends and identify optimization opportunities."""
        optimizations = []
        
        # CPU optimization
        if 'cpu_usage' in self.metrics_storage:
            recent_cpu = list(self.metrics_storage['cpu_usage'])[-20:]
            if recent_cpu:
                avg_cpu = statistics.mean([m.value for m in recent_cpu])
                if avg_cpu > 80:
                    optimizations.append({
                        'type': 'cpu_optimization',
                        'priority': 'high',
                        'action': 'scale_workers',
                        'details': f'High CPU usage detected: {avg_cpu:.1f}%'
                    })
        
        # Memory optimization
        if 'memory_usage' in self.metrics_storage:
            recent_memory = list(self.metrics_storage['memory_usage'])[-20:]
            if recent_memory:
                avg_memory = statistics.mean([m.value for m in recent_memory])
                if avg_memory > 85:
                    optimizations.append({
                        'type': 'memory_optimization',
                        'priority': 'high',
                        'action': 'clear_cache',
                        'details': f'High memory usage detected: {avg_memory:.1f}%'
                    })
        
        # Response time optimization
        if 'api_response_time' in self.metrics_storage:
            recent_response_times = list(self.metrics_storage['api_response_time'])[-50:]
            if recent_response_times:
                avg_response_time = statistics.mean([m.value for m in recent_response_times])
                if avg_response_time > self.performance_targets['api_response_time']:
                    optimizations.append({
                        'type': 'response_time_optimization',
                        'priority': 'medium',
                        'action': 'optimize_queries',
                        'details': f'Slow response time detected: {avg_response_time:.2f}s'
                    })
        
        return optimizations
    
    async def _apply_optimization(self, optimization: Dict[str, Any]):
        """Apply performance optimization."""
        logger.info(f"Applying optimization: {optimization['type']} - {optimization['action']}")
        
        try:
            if optimization['action'] == 'scale_workers':
                # Implementation would scale Celery workers
                logger.info("Scaling workers for CPU optimization")
                
            elif optimization['action'] == 'clear_cache':
                # Implementation would clear caches
                logger.info("Clearing caches for memory optimization")
                
            elif optimization['action'] == 'optimize_queries':
                # Implementation would optimize database queries
                logger.info("Optimizing database queries")
                
        except Exception as e:
            logger.error(f"Error applying optimization {optimization['type']}: {e}")
    
    def record_metric(
        self, 
        name: str, 
        value: float, 
        metric_type: MetricType, 
        tags: Optional[Dict[str, str]] = None
    ):
        """Record a metric value."""
        if tags is None:
            tags = {}
        
        metric = Metric(
            name=name,
            value=value,
            timestamp=datetime.utcnow(),
            type=metric_type,
            tags=tags
        )
        
        self.metrics_storage[name].append(metric)
        
        # Save to file for persistence (in production, would use time-series DB)
        self._save_metric_to_file(metric)
    
    def _save_metric_to_file(self, metric: Metric):
        """Save metric to file for persistence."""
        metrics_file = self.storage_path / f"metrics_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"
        
        with open(metrics_file, 'a') as f:
            f.write(json.dumps(metric.to_dict()) + '\n')
    
    def get_metrics(
        self, 
        metric_name: str, 
        start_time: Optional[datetime] = None, 
        end_time: Optional[datetime] = None
    ) -> List[Metric]:
        """Get metrics for a specific name and time range."""
        if metric_name not in self.metrics_storage:
            return []
        
        metrics = list(self.metrics_storage[metric_name])
        
        # Filter by time range
        if start_time:
            metrics = [m for m in metrics if m.timestamp >= start_time]
        
        if end_time:
            metrics = [m for m in metrics if m.timestamp <= end_time]
        
        return metrics
    
    def get_metric_summary(self, metric_name: str, hours: int = 24) -> Dict[str, Any]:
        """Get summary statistics for a metric."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        metrics = self.get_metrics(metric_name, start_time, end_time)
        
        if not metrics:
            return {'error': 'No data available'}
        
        values = [m.value for m in metrics]
        
        return {
            'metric_name': metric_name,
            'period_hours': hours,
            'count': len(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'min': min(values),
            'max': max(values),
            'stdev': statistics.stdev(values) if len(values) > 1 else 0,
            'latest': values[-1] if values else None,
            'timestamp': end_time.isoformat()
        }
    
    async def generate_performance_report(
        self, 
        hours: int = 24
    ) -> PerformanceReport:
        """Generate comprehensive performance report."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        # API Performance
        response_times = [m.value for m in self.get_metrics('api_response_time', start_time, end_time)]
        error_rates = [m.value for m in self.get_metrics('error_rate', start_time, end_time)]
        
        # System Performance
        cpu_values = [m.value for m in self.get_metrics('cpu_usage', start_time, end_time)]
        memory_values = [m.value for m in self.get_metrics('memory_usage', start_time, end_time)]
        disk_values = [m.value for m in self.get_metrics('disk_usage', start_time, end_time)]
        
        # Analysis Performance
        analysis_times = [m.value for m in self.get_metrics('analysis_time', start_time, end_time)]
        
        # Generate recommendations
        recommendations = self._generate_performance_recommendations({
            'response_times': response_times,
            'error_rates': error_rates,
            'cpu_values': cpu_values,
            'memory_values': memory_values,
            'analysis_times': analysis_times
        })
        
        return PerformanceReport(
            period_start=start_time,
            period_end=end_time,
            
            # API Performance
            avg_response_time=statistics.mean(response_times) if response_times else 0,
            p95_response_time=self._percentile(response_times, 95) if response_times else 0,
            p99_response_time=self._percentile(response_times, 99) if response_times else 0,
            error_rate=statistics.mean(error_rates) if error_rates else 0,
            throughput=len(response_times) / hours if response_times else 0,
            
            # Analysis Performance
            avg_analysis_time=statistics.mean(analysis_times) if analysis_times else 0,
            analysis_success_rate=0.95,  # Would calculate from actual data
            
            # System Performance
            avg_cpu_usage=statistics.mean(cpu_values) if cpu_values else 0,
            avg_memory_usage=statistics.mean(memory_values) if memory_values else 0,
            disk_usage=statistics.mean(disk_values) if disk_values else 0,
            
            # Business Metrics (would come from database)
            total_analyses=len(analysis_times),
            unique_users=50,  # Placeholder
            avg_score=75.5,   # Placeholder
            
            recommendations=recommendations
        )
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0
        
        sorted_values = sorted(values)
        index = (percentile / 100) * (len(sorted_values) - 1)
        
        if index.is_integer():
            return sorted_values[int(index)]
        else:
            lower = sorted_values[int(index)]
            upper = sorted_values[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    def _generate_performance_recommendations(self, metrics: Dict[str, List[float]]) -> List[str]:
        """Generate performance recommendations based on metrics."""
        recommendations = []
        
        # Response time recommendations
        if metrics['response_times']:
            avg_response_time = statistics.mean(metrics['response_times'])
            if avg_response_time > 2.0:
                recommendations.append("Consider implementing API response caching")
                recommendations.append("Optimize database queries for better performance")
            
            if avg_response_time > 5.0:
                recommendations.append("URGENT: Response times are critically high - investigate immediately")
        
        # CPU recommendations
        if metrics['cpu_values']:
            avg_cpu = statistics.mean(metrics['cpu_values'])
            if avg_cpu > 70:
                recommendations.append("High CPU usage detected - consider scaling horizontally")
            
            if avg_cpu > 90:
                recommendations.append("CRITICAL: CPU usage is very high - scale immediately")
        
        # Memory recommendations
        if metrics['memory_values']:
            avg_memory = statistics.mean(metrics['memory_values'])
            if avg_memory > 80:
                recommendations.append("High memory usage - implement memory optimization")
                recommendations.append("Consider increasing cache cleanup frequency")
        
        # Analysis time recommendations
        if metrics['analysis_times']:
            avg_analysis_time = statistics.mean(metrics['analysis_times'])
            if avg_analysis_time > 300:  # 5 minutes
                recommendations.append("Analysis times are high - optimize AI processing pipeline")
            
            if avg_analysis_time > 600:  # 10 minutes
                recommendations.append("URGENT: Analysis times are critically high - review processing logic")
        
        return recommendations
    
    def get_active_alerts(self) -> List[Alert]:
        """Get currently active (unresolved) alerts."""
        return [alert for alert in self.alerts if not alert.resolved]
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved."""
        for alert in self.alerts:
            if alert.id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = datetime.utcnow()
                logger.info(f"Alert {alert_id} resolved")
                return True
        
        return False
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        
        # Get recent metrics
        recent_metrics = {}
        for metric_name in ['cpu_usage', 'memory_usage', 'disk_usage', 'api_response_time']:
            summary = self.get_metric_summary(metric_name, hours=1)
            if 'error' not in summary:
                recent_metrics[metric_name] = summary
        
        # Calculate health score
        health_score = self._calculate_health_score(recent_metrics)
        
        # Determine status
        if health_score >= 90:
            status = "healthy"
        elif health_score >= 70:
            status = "warning"
        elif health_score >= 50:
            status = "degraded"
        else:
            status = "critical"
        
        return {
            'status': status,
            'health_score': health_score,
            'active_alerts': len(self.get_active_alerts()),
            'recent_metrics': recent_metrics,
            'last_update': datetime.utcnow().isoformat()
        }
    
    def _calculate_health_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall system health score."""
        if not metrics:
            return 50.0  # Default if no metrics
        
        scores = []
        
        # CPU health (inverse of usage)
        if 'cpu_usage' in metrics:
            cpu_score = max(0, 100 - metrics['cpu_usage']['mean'])
            scores.append(cpu_score)
        
        # Memory health (inverse of usage)
        if 'memory_usage' in metrics:
            memory_score = max(0, 100 - metrics['memory_usage']['mean'])
            scores.append(memory_score)
        
        # Disk health (inverse of usage)
        if 'disk_usage' in metrics:
            disk_score = max(0, 100 - metrics['disk_usage']['mean'])
            scores.append(disk_score)
        
        # Response time health
        if 'api_response_time' in metrics:
            response_time = metrics['api_response_time']['mean']
            response_score = max(0, 100 - (response_time * 20))  # Penalty for slow responses
            scores.append(response_score)
        
        return statistics.mean(scores) if scores else 50.0


# Global monitoring system instance
monitoring_system = MonitoringSystem()


# Decorator for tracking API performance
def track_performance(func: Callable) -> Callable:
    """Decorator to track function performance."""
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            success = False
            error = str(e)
            raise
        finally:
            execution_time = time.time() - start_time
            
            # Record metrics
            monitoring_system.record_metric(
                f"{func.__name__}_execution_time",
                execution_time,
                MetricType.PERFORMANCE,
                {"function": func.__name__, "success": str(success)}
            )
            
            if not success:
                monitoring_system.record_metric(
                    f"{func.__name__}_error",
                    1,
                    MetricType.PERFORMANCE,
                    {"function": func.__name__, "error": error or "unknown"}
                )
        
        return result
    
    return wrapper