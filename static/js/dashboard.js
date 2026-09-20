/**
 * Dashboard & Interactive Sentiment Analytics Client Script
 * Handles real-time single comment inference, KPI live updates,
 * and Chart.js dynamic visualizations.
 */

let distributionChartInstance = null;
let domainChartInstance = null;
let benchmarkChartInstance = null;

document.addEventListener('DOMContentLoaded', () => {
    // 1. Initialize Chart.js visualizations
    if (typeof initialStats !== 'undefined' && initialStats) {
        initDistributionChart(initialStats);
        initDomainChart(initialStats);
    }
    if (typeof initialMetrics !== 'undefined' && initialMetrics) {
        initBenchmarkChart(initialMetrics);
    }

    // 2. Setup Form Submission Listener
    const commentForm = document.getElementById('commentForm');
    if (commentForm) {
        commentForm.addEventListener('submit', handleCommentSubmission);
    }
});

// Tab Switcher
function switchTab(tabName) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

    const activeBtn = Array.from(document.querySelectorAll('.tab-btn')).find(
        btn => btn.getAttribute('onclick').includes(tabName)
    );
    if (activeBtn) activeBtn.classList.add('active');

    const targetContent = document.getElementById(`tab-${tabName}`);
    if (targetContent) {
        targetContent.classList.add('active');
        // Trigger resize on charts to prevent rendering artifacts
        if (tabName === 'distribution' && distributionChartInstance) distributionChartInstance.resize();
        if (tabName === 'domains' && domainChartInstance) domainChartInstance.resize();
        if (tabName === 'benchmarks' && benchmarkChartInstance) benchmarkChartInstance.resize();
    }
}

// Handle Single Comment Submission
async function handleCommentSubmission(e) {
    e.preventDefault();

    const commentInput = document.getElementById('commentText');
    const domainInput = document.getElementById('domainSelect');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const btnText = analyzeBtn.querySelector('.btn-text');
    const spinner = analyzeBtn.querySelector('.spinner');

    const comment = commentInput.value.trim();
    const domain = domainInput.value;

    if (!comment) return;

    // Loading State
    analyzeBtn.disabled = true;
    btnText.textContent = 'Analyzing...';
    spinner.style.display = 'inline-block';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify({ comment, domain })
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
            alert('Analysis Error: ' + (data.error || 'Failed to process comment'));
            return;
        }

        // Render Prediction Result
        displayPredictionResult(data);

        // Update live KPIs & Table
        if (data.db_saved) {
            updateKPIs(data.sentiment);
            prependRecentComment(data);
            refreshCharts();
        } else if (data.db_warning) {
            console.warn('DB Notice:', data.db_warning);
        }

    } catch (err) {
        console.error('Submission error:', err);
        alert('Network or server error while connecting to Flask backend.');
    } finally {
        analyzeBtn.disabled = false;
        btnText.textContent = 'Analyze Sentiment';
        spinner.style.display = 'none';
    }
}

// Display Result Card
function displayPredictionResult(data) {
    const card = document.getElementById('predictionResult');
    const badge = document.getElementById('sentimentBadge');
    const confScore = document.getElementById('confidenceScore');
    const confPercent = document.getElementById('confidencePercent');
    const confProgress = document.getElementById('confidenceProgress');
    const preprocessed = document.getElementById('preprocessedText');
    const timestamp = document.getElementById('resultTimestamp');

    // VADER elements
    const vaderBadge = document.getElementById('vaderBadge');
    const vaderCompound = document.getElementById('vaderCompound');
    const vaderPos = document.getElementById('vaderPos');
    const vaderNeg = document.getElementById('vaderNeg');
    const vaderNeu = document.getElementById('vaderNeu');

    // Sentiment badge
    const sentiment = data.sentiment.toLowerCase();
    badge.className = `sentiment-badge ${sentiment}`;
    badge.textContent = sentiment.toUpperCase();

    // Confidence
    const conf = Math.round(data.confidence * 100);
    confScore.textContent = `${conf}% Confidence`;
    confPercent.textContent = `${conf}%`;
    confProgress.className = `progress-fill ${sentiment}`;
    confProgress.style.width = `${conf}%`;

    // Preprocessed text
    preprocessed.textContent = data.preprocessed_comment || '(empty)';
    timestamp.textContent = new Date().toLocaleTimeString();

    // VADER Baseline
    if (data.vader) {
        const vs = data.vader.sentiment.toLowerCase();
        vaderBadge.className = `vader-badge ${vs}`;
        vaderBadge.textContent = vs.toUpperCase();
        vaderCompound.textContent = data.vader.compound;
        vaderPos.textContent = data.vader.positive;
        vaderNeg.textContent = data.vader.negative;
        vaderNeu.textContent = data.vader.neutral;
    }

    card.style.display = 'block';
    card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Update KPI DOM counts
function updateKPIs(sentiment) {
    const totalEl = document.getElementById('kpi-total');
    const targetEl = document.getElementById(`kpi-${sentiment}`);
    const sessionCountEl = document.getElementById('sessionCommentsCount');

    if (totalEl) {
        const curTotal = parseInt(totalEl.textContent) || 0;
        totalEl.textContent = curTotal + 1;
    }
    if (sessionCountEl) {
        const curSess = parseInt(sessionCountEl.textContent) || 0;
        sessionCountEl.textContent = curSess + 1;
    }
    if (targetEl) {
        const curTarget = parseInt(targetEl.textContent) || 0;
        targetEl.textContent = curTarget + 1;
    }
}

// Prepend to Recent Comments Table
function prependRecentComment(data) {
    const tbody = document.getElementById('recentCommentsBody');
    if (!tbody) return;

    // Remove empty placeholder row if exists
    const emptyRow = tbody.querySelector('.empty-state');
    if (emptyRow) emptyRow.closest('tr').remove();

    const tr = document.createElement('tr');
    tr.innerHTML = `
        <td>#${data.id || 'new'}</td>
        <td class="comment-cell" title="${data.comment}">${data.comment}</td>
        <td><span class="domain-tag">${data.domain}</span></td>
        <td><span class="badge badge-${data.sentiment}">${data.sentiment.charAt(0).toUpperCase() + data.sentiment.slice(1)}</span></td>
        <td>${Math.round(data.confidence * 100)}%</td>
        <td><span class="vader-chip">${data.vader ? data.vader.sentiment : '-'} (${data.vader ? data.vader.compound : '0'})</span></td>
        <td class="time-cell">Just now</td>
    `;
    tbody.insertBefore(tr, tbody.firstChild);
}

// Chart 1: Sentiment Distribution Doughnut
function initDistributionChart(stats) {
    const ctx = document.getElementById('distributionChart');
    if (!ctx) return;

    const total = stats.total || 0;
    const hasData = total > 0;
    const chartData = hasData
        ? [stats.positive || 0, stats.negative || 0, stats.neutral || 0]
        : [0, 0, 0];

    const data = {
        labels: ['Positive', 'Negative', 'Neutral / Borderline'],
        datasets: [{
            data: chartData,
            backgroundColor: ['#10b981', '#ef4444', '#6366f1'],
            borderWidth: 2,
            borderColor: '#ffffff',
            hoverOffset: 4
        }]
    };

    distributionChartInstance = new Chart(ctx, {
        type: 'doughnut',
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { font: { family: 'Inter', size: 12 } }
                },
                title: {
                    display: true,
                    text: hasData ? 'Session Sentiment Breakdown' : 'Session Sentiment Breakdown (Awaiting Comments)',
                    font: { family: 'Inter', size: 14, weight: '600' }
                }
            },
            cutout: '65%'
        }
    });
}

// Chart 2: Domain-wise Stacked Bar
function initDomainChart(stats) {
    const ctx = document.getElementById('domainChart');
    if (!ctx) return;

    // Aggregate by domain
    const domains = {};
    if (stats && stats.domain_distribution) {
        stats.domain_distribution.forEach(row => {
            const dom = row.domain || 'general';
            if (!domains[dom]) domains[dom] = { positive: 0, negative: 0, neutral: 0 };
            domains[dom][row.sentiment.toLowerCase()] = row.count;
        });
    }

    const hasDomains = Object.keys(domains).length > 0;
    const labels = hasDomains ? Object.keys(domains) : ['GENERAL'];
    const posData = labels.map(d => (domains[d] ? domains[d].positive : 0));
    const negData = labels.map(d => (domains[d] ? domains[d].negative : 0));
    const neuData = labels.map(d => (domains[d] ? domains[d].neutral : 0));

    domainChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels.map(l => l.replace('_', ' ').toUpperCase()),
            datasets: [
                { label: 'Positive', data: posData, backgroundColor: '#10b981' },
                { label: 'Negative', data: negData, backgroundColor: '#ef4444' },
                { label: 'Neutral', data: neuData, backgroundColor: '#6366f1' }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { stacked: true, ticks: { font: { family: 'Inter', size: 10 } } },
                y: { stacked: true, beginAtZero: true, ticks: { precision: 0 } }
            },
            plugins: {
                legend: { position: 'bottom' },
                title: {
                    display: true,
                    text: hasDomains ? 'Sentiment by Healthcare Domain' : 'Sentiment by Healthcare Domain (Awaiting Comments)',
                    font: { family: 'Inter', size: 14, weight: '600' }
                }
            }
        }
    });
}

// Chart 3: Model Benchmark Comparisons
function initBenchmarkChart(metrics) {
    const ctx = document.getElementById('benchmarkChart');
    if (!ctx || !metrics.models_comparison) return;

    const models = Object.keys(metrics.models_comparison);
    const accuracies = models.map(m => metrics.models_comparison[m].accuracy);
    const precisions = models.map(m => metrics.models_comparison[m].precision);
    const recalls = models.map(m => metrics.models_comparison[m].recall);
    const f1s = models.map(m => metrics.models_comparison[m].f1_score);

    benchmarkChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: models,
            datasets: [
                { label: 'Accuracy %', data: accuracies, backgroundColor: '#0284c7' },
                { label: 'Precision %', data: precisions, backgroundColor: '#14b8a6' },
                { label: 'Recall %', data: recalls, backgroundColor: '#f59e0b' },
                { label: 'F1-Score %', data: f1s, backgroundColor: '#8b5cf6' }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { min: 80, max: 100, ticks: { callback: v => v + '%' } }
            },
            plugins: {
                legend: { position: 'bottom' },
                title: {
                    display: true,
                    text: 'Empirical Test Performance Comparison',
                    font: { family: 'Inter', size: 14, weight: '600' }
                }
            }
        }
    });
}

// Refresh Charts dynamically via /api/stats
async function refreshCharts() {
    try {
        const res = await fetch('/api/stats');
        const stats = await res.json();
        if (!stats) return;

        // Sync KPI numbers
        if (typeof stats.total !== 'undefined') {
            const totalEl = document.getElementById('kpi-total');
            const sessEl = document.getElementById('sessionCommentsCount');
            const posEl = document.getElementById('kpi-positive');
            const negEl = document.getElementById('kpi-negative');
            const neuEl = document.getElementById('kpi-neutral');
            const posPct = document.getElementById('kpi-positive-pct');
            const negPct = document.getElementById('kpi-negative-pct');

            if (totalEl) totalEl.textContent = stats.total;
            if (sessEl) sessEl.textContent = stats.total;
            if (posEl) posEl.textContent = stats.positive || 0;
            if (negEl) negEl.textContent = stats.negative || 0;
            if (neuEl) neuEl.textContent = stats.neutral || 0;
            if (posPct) posPct.textContent = `${stats.positive_pct || 0}% of total`;
            if (negPct) negPct.textContent = `${stats.negative_pct || 0}% of total`;
        }

        if (distributionChartInstance) {
            distributionChartInstance.data.datasets[0].data = [
                stats.positive || 0,
                stats.negative || 0,
                stats.neutral || 0
            ];
            if (stats.total > 0 && distributionChartInstance.options.plugins.title) {
                distributionChartInstance.options.plugins.title.text = 'Session Sentiment Breakdown';
            }
            distributionChartInstance.update();
        }

        if (domainChartInstance && stats.domain_distribution) {
            const domains = {};
            stats.domain_distribution.forEach(row => {
                const dom = row.domain || 'general';
                if (!domains[dom]) domains[dom] = { positive: 0, negative: 0, neutral: 0 };
                domains[dom][row.sentiment.toLowerCase()] = row.count;
            });
            const labels = Object.keys(domains);
            if (labels.length > 0) {
                domainChartInstance.data.labels = labels.map(l => l.replace('_', ' ').toUpperCase());
                domainChartInstance.data.datasets[0].data = labels.map(d => domains[d].positive);
                domainChartInstance.data.datasets[1].data = labels.map(d => domains[d].negative);
                domainChartInstance.data.datasets[2].data = labels.map(d => domains[d].neutral);
                if (domainChartInstance.options.plugins.title) {
                    domainChartInstance.options.plugins.title.text = 'Sentiment by Healthcare Domain';
                }
                domainChartInstance.update();
            }
        }
    } catch (e) {
        console.warn('Chart refresh warning:', e);
    }
}
