/**
 * Advanced Analytics Dashboard Script
 * Handles Chart.js visualization for sentiment timeline trends and domain distributions.
 */

document.addEventListener('DOMContentLoaded', function () {
    const payloadEl = document.getElementById('analytics-payload-data');
    if (!payloadEl) return;

    let payload = {};
    try {
        payload = JSON.parse(payloadEl.textContent || '{}');
    } catch (e) {
        console.warn('Failed to parse analytics payload JSON:', e);
        return;
    }

    const summary = payload.summary || {};
    if (!summary.total || summary.total <= 0) {
        return; // Empty state handles display
    }

    // -------------------------------------------------------------
    // Chart 1: Sentiment Trend Over Time (Line Chart)
    // -------------------------------------------------------------
    const trendCtx = document.getElementById('sentimentTrendChart');
    const trendData = payload.sentiment_trend || {};

    if (trendCtx && trendData.labels && trendData.labels.length > 0) {
        new Chart(trendCtx, {
            type: 'line',
            data: {
                labels: trendData.labels,
                datasets: [
                    {
                        label: 'Positive Feedback',
                        data: trendData.positive || [],
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.12)',
                        fill: true,
                        tension: 0.3,
                        borderWidth: 2.5,
                        pointBackgroundColor: '#10b981',
                        pointRadius: 4,
                        pointHoverRadius: 6
                    },
                    {
                        label: 'Negative Feedback',
                        data: trendData.negative || [],
                        borderColor: '#ef4444',
                        backgroundColor: 'rgba(239, 68, 68, 0.12)',
                        fill: true,
                        tension: 0.3,
                        borderWidth: 2.5,
                        pointBackgroundColor: '#ef4444',
                        pointRadius: 4,
                        pointHoverRadius: 6
                    },
                    {
                        label: 'Neutral Feedback',
                        data: trendData.neutral || [],
                        borderColor: '#6366f1',
                        backgroundColor: 'rgba(99, 102, 241, 0.10)',
                        fill: true,
                        tension: 0.3,
                        borderWidth: 2,
                        pointBackgroundColor: '#6366f1',
                        pointRadius: 3,
                        pointHoverRadius: 5
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false
                },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: { font: { family: 'Inter', size: 11 } }
                    },
                    y: {
                        beginAtZero: true,
                        ticks: { precision: 0, font: { family: 'Inter', size: 11 } },
                        grid: { color: 'rgba(226, 232, 240, 0.7)' }
                    }
                },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: { font: { family: 'Inter', size: 12 }, boxWidth: 14 }
                    },
                    tooltip: {
                        callbacks: {
                            afterTitle: function (items) {
                                let total = 0;
                                items.forEach(it => { total += (it.parsed.y || 0); });
                                return 'Total in period: ' + total;
                            }
                        }
                    }
                }
            }
        });
    }

    // -------------------------------------------------------------
    // Chart 2: Domain-wise Sentiment Comparison (Stacked Bar Chart)
    // -------------------------------------------------------------
    const domainCtx = document.getElementById('domainAnalyticsChart');
    const domains = payload.domain_distribution || [];

    if (domainCtx && domains.length > 0) {
        const domainLabels = domains.map(d => (d.domain || 'general').replace(/_/g, ' ').toUpperCase());
        const posDomainData = domains.map(d => d.positive || 0);
        const negDomainData = domains.map(d => d.negative || 0);
        const neuDomainData = domains.map(d => d.neutral || 0);

        new Chart(domainCtx, {
            type: 'bar',
            data: {
                labels: domainLabels,
                datasets: [
                    {
                        label: 'Positive',
                        data: posDomainData,
                        backgroundColor: '#10b981',
                        borderRadius: 3
                    },
                    {
                        label: 'Negative',
                        data: negDomainData,
                        backgroundColor: '#ef4444',
                        borderRadius: 3
                    },
                    {
                        label: 'Neutral',
                        data: neuDomainData,
                        backgroundColor: '#6366f1',
                        borderRadius: 3
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        stacked: true,
                        grid: { display: false },
                        ticks: { font: { family: 'Inter', size: 10, weight: '500' } }
                    },
                    y: {
                        stacked: true,
                        beginAtZero: true,
                        ticks: { precision: 0, font: { family: 'Inter', size: 11 } },
                        grid: { color: 'rgba(226, 232, 240, 0.7)' }
                    }
                },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: { font: { family: 'Inter', size: 12 }, boxWidth: 14 }
                    },
                    tooltip: {
                        callbacks: {
                            afterTitle: function (items) {
                                const idx = items[0].dataIndex;
                                const item = domains[idx] || {};
                                return 'Specialty Volume: ' + (item.total || 0) + ' comments';
                            }
                        }
                    }
                }
            }
        });
    }
});
