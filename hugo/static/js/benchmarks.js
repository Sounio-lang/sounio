/**
 * Sounio Benchmark Visualizations
 * Interactive charts using Chart.js for scientific credibility
 */

(function() {
    'use strict';

    // Color palette matching site theme
    const COLORS = {
        cpu: '#495057',
        gpu_ptx: '#fa5252',
        gpu_metal: '#228be6',
        real_valued: '#868e96',
        quaternion: '#7950f2',
        octonion: '#fd7e14',
        success: '#2f9e44',
        llvm: '#228be6',
        cranelift: '#fd7e14',
        native: '#2f9e44',
        ptx: '#fa5252',
        metal: '#7950f2'
    };

    const FONT = {
        family: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif",
        size: 12
    };

    // Chart 1: GPU Speedup (Octonion Operations)
    function renderGPUSpeedupChart() {
        const ctx = document.getElementById('gpu-speedup-chart');
        if (!ctx) return;

        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Multiply', 'Add', 'Norm', 'Conjugate'],
                datasets: [
                    {
                        label: 'CPU (AMD Ryzen 9 7950X)',
                        data: [8.5, 12.3, 9.7, 11.2],
                        backgroundColor: COLORS.cpu,
                        borderRadius: 4
                    },
                    {
                        label: 'GPU PTX (NVIDIA RTX 4090)',
                        data: [142.7, 189.4, 156.8, 178.3],
                        backgroundColor: COLORS.gpu_ptx,
                        borderRadius: 4
                    },
                    {
                        label: 'GPU Metal (Apple M2 Ultra)',
                        data: [156.3, 201.7, 168.2, 192.5],
                        backgroundColor: COLORS.gpu_metal,
                        borderRadius: 4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Octonion Operation Throughput (GFLOPS)',
                        font: { ...FONT, size: 16, weight: '600' },
                        padding: { bottom: 20 }
                    },
                    legend: {
                        position: 'bottom',
                        labels: { font: FONT, padding: 15, usePointStyle: true }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const val = context.parsed.y;
                                const speedup = context.datasetIndex > 0
                                    ? ' (' + (val / context.chart.data.datasets[0].data[context.dataIndex]).toFixed(1) + '× speedup)'
                                    : ' (baseline)';
                                return context.dataset.label + ': ' + val + ' GFLOPS' + speedup;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'GFLOPS (higher is better)',
                            font: FONT
                        },
                        grid: { color: '#f1f3f5' }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Octonion Operation',
                            font: FONT
                        },
                        grid: { display: false }
                    }
                }
            }
        });
    }

    // Chart 2: Parameter Efficiency (ONN vs Real-Valued)
    function renderParameterEfficiencyChart() {
        const ctx = document.getElementById('parameter-efficiency-chart');
        if (!ctx) return;

        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['MNIST', 'CIFAR-10'],
                datasets: [
                    {
                        label: 'Real-Valued',
                        data: [1000000, 5000000],
                        backgroundColor: COLORS.real_valued,
                        borderRadius: 4
                    },
                    {
                        label: 'Quaternion (4× compression)',
                        data: [250000, 1250000],
                        backgroundColor: COLORS.quaternion,
                        borderRadius: 4
                    },
                    {
                        label: 'Octonion (8× compression)',
                        data: [125000, 625000],
                        backgroundColor: COLORS.octonion,
                        borderRadius: 4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Parameter Count by Network Type',
                        font: { ...FONT, size: 16, weight: '600' },
                        padding: { bottom: 20 }
                    },
                    legend: {
                        position: 'bottom',
                        labels: { font: FONT, padding: 15, usePointStyle: true }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const val = context.parsed.y;
                                return context.dataset.label + ': ' + val.toLocaleString() + ' parameters';
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Parameters (fewer is better)',
                            font: FONT
                        },
                        ticks: {
                            callback: function(val) {
                                if (val >= 1000000) return (val / 1000000).toFixed(1) + 'M';
                                if (val >= 1000) return (val / 1000).toFixed(0) + 'K';
                                return val;
                            }
                        },
                        grid: { color: '#f1f3f5' }
                    },
                    x: { grid: { display: false } }
                }
            }
        });
    }

    // Chart 3: Backend Comparison Radar
    function renderBackendRadarChart() {
        const ctx = document.getElementById('backend-radar-chart');
        if (!ctx) return;

        new Chart(ctx, {
            type: 'radar',
            data: {
                labels: ['Compile Speed', 'Runtime Perf', 'Binary Size', 'Optimization', 'GPU Support'],
                datasets: [
                    {
                        label: 'LLVM',
                        data: [40, 95, 75, 100, 60],
                        borderColor: COLORS.llvm,
                        backgroundColor: COLORS.llvm + '20',
                        borderWidth: 2,
                        pointBackgroundColor: COLORS.llvm
                    },
                    {
                        label: 'Cranelift JIT',
                        data: [100, 78, 55, 60, 40],
                        borderColor: COLORS.cranelift,
                        backgroundColor: COLORS.cranelift + '20',
                        borderWidth: 2,
                        pointBackgroundColor: COLORS.cranelift
                    },
                    {
                        label: 'Native',
                        data: [65, 92, 100, 80, 50],
                        borderColor: COLORS.native,
                        backgroundColor: COLORS.native + '20',
                        borderWidth: 2,
                        pointBackgroundColor: COLORS.native
                    },
                    {
                        label: 'PTX (NVIDIA)',
                        data: [55, 100, 90, 95, 100],
                        borderColor: COLORS.ptx,
                        backgroundColor: COLORS.ptx + '20',
                        borderWidth: 2,
                        pointBackgroundColor: COLORS.ptx
                    },
                    {
                        label: 'Metal (Apple)',
                        data: [60, 98, 92, 90, 100],
                        borderColor: COLORS.metal,
                        backgroundColor: COLORS.metal + '20',
                        borderWidth: 2,
                        pointBackgroundColor: COLORS.metal
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Backend Comparison (5 Codegen Targets)',
                        font: { ...FONT, size: 16, weight: '600' },
                        padding: { bottom: 20 }
                    },
                    legend: {
                        position: 'bottom',
                        labels: { font: FONT, padding: 12, usePointStyle: true }
                    }
                },
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 100,
                        ticks: { stepSize: 25, display: false },
                        grid: { color: '#dee2e6' },
                        pointLabels: { font: { ...FONT, size: 11 } }
                    }
                }
            }
        });
    }

    // Chart 4: Moufang Validation Throughput
    function renderMoufangChart() {
        const ctx = document.getElementById('moufang-chart');
        if (!ctx) return;

        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['CPU\n(Ryzen 9 7950X)', 'GPU PTX\n(RTX 4090)', 'GPU Metal\n(M2 Ultra)'],
                datasets: [{
                    label: 'Tests per Second',
                    data: [76337, 1541850, 1391649],
                    backgroundColor: [COLORS.cpu, COLORS.gpu_ptx, COLORS.gpu_metal],
                    borderRadius: 4,
                    barThickness: 60
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                indexAxis: 'y',
                plugins: {
                    title: {
                        display: true,
                        text: 'Moufang Identity Validation Throughput',
                        font: { ...FONT, size: 16, weight: '600' },
                        padding: { bottom: 20 }
                    },
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const val = context.parsed.x;
                                const speedup = val > 76337 ? ' (' + (val / 76337).toFixed(1) + '× vs CPU)' : ' (baseline)';
                                return val.toLocaleString() + ' tests/sec' + speedup;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Tests per Second (higher is better)',
                            font: FONT
                        },
                        ticks: {
                            callback: function(val) {
                                if (val >= 1000000) return (val / 1000000).toFixed(1) + 'M';
                                if (val >= 1000) return (val / 1000).toFixed(0) + 'K';
                                return val;
                            }
                        },
                        grid: { color: '#f1f3f5' }
                    },
                    y: { grid: { display: false } }
                }
            }
        });
    }

    // Initialize all charts
    function initCharts() {
        renderGPUSpeedupChart();
        renderParameterEfficiencyChart();
        renderBackendRadarChart();
        renderMoufangChart();
    }

    // Expose globally for deferred initialization
    window.initBenchmarkCharts = initCharts;

    // Also auto-init if Chart is already available
    if (typeof Chart !== 'undefined') {
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', initCharts);
        } else {
            initCharts();
        }
    }

})();
